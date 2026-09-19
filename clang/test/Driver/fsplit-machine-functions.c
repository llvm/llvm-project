// RUN: rm -rf %t && mkdir %t
// RUN: llvm-profdata merge -o %t/default.profdata %S/Inputs/a.proftext

// RUN: %clang -### --target=x86_64 -fprofile-use=%t/default.profdata -fsplit-machine-functions %s 2>&1 | FileCheck %s --check-prefixes=CHECK,LTO-NEG
// RUN: %clang -### --target=aarch64 -fprofile-use=%t/default.profdata -fsplit-machine-functions %s 2>&1 | FileCheck %s --check-prefixes=CHECK,LTO-NEG
// RUN: %clang -### --target=x86_64 -fprofile-use=%t/default.profdata -fsplit-machine-functions -fno-split-machine-functions %s -c 2>&1 | FileCheck -check-prefix=NEG %s

// CHECK:      "-fsplit-machine-functions=all"
// CHECK-SAME: "-fprofile-instrument-use=llvm"
// CHECK-SAME: "-fprofile-instrument-use-path={{.*}}default.profdata"

// NEG-NOT:    "-fsplit-machine-functions=all"
// NEG:        "-fsplit-machine-functions=none"

/// The mode spellings are forwarded to -cc1 verbatim and the last one wins.
// RUN: %clang -### -c --target=x86_64 -fsplit-machine-functions=bbsections %s 2>&1 | FileCheck %s --check-prefix=BBSECTIONS
// RUN: %clang -### -c --target=x86_64 -fsplit-machine-functions -fsplit-machine-functions=bbsections %s 2>&1 | FileCheck %s --check-prefix=BBSECTIONS
// BBSECTIONS: "-fsplit-machine-functions=bbsections"

// RUN: %clang -### -c --target=x86_64 -fsplit-machine-functions=none %s 2>&1 | FileCheck %s --check-prefix=NONE
// RUN: %clang -### -c --target=x86_64 -fsplit-machine-functions -fno-split-machine-functions %s 2>&1 | FileCheck %s --check-prefix=NONE
// NONE: "-fsplit-machine-functions=none"

// RUN: %clang -### -c --target=x86_64 -fno-split-machine-functions -fsplit-machine-functions %s 2>&1 | FileCheck %s --check-prefix=ALL
// RUN: %clang -### -c --target=x86_64 -fsplit-machine-functions=all %s 2>&1 | FileCheck %s --check-prefix=ALL
// ALL: "-fsplit-machine-functions=all"

/// Nothing is forwarded when the option is absent.
// RUN: %clang -### -c --target=x86_64 %s 2>&1 | FileCheck %s --check-prefix=ABSENT
// ABSENT-NOT: "-fsplit-machine-functions

// RUN: %clang -### --target=x86_64-linux -flto -fsplit-machine-functions %s 2>&1 | FileCheck %s --check-prefix=LTO
// RUN: %clang -### --target=x86_64-linux -flto -fsplit-machine-functions -fno-split-machine-functions %s 2>&1 | FileCheck %s --check-prefix=LTO-NEG

// LTO:         "-plugin-opt=-split-machine-functions"
// LTO-NEG-NOT: "-plugin-opt=-split-machine-functions"

// RUN: not %clang -### -c --target=arm-unknown-linux -fsplit-machine-functions %s 2>&1 | FileCheck %s --check-prefix=ERR
// RUN: not %clang -### -c --target=arm-unknown-linux -fsplit-machine-functions=bbsections %s 2>&1 | FileCheck %s --check-prefix=ERR-BBSECTIONS
// ERR: error: unsupported option '-fsplit-machine-functions' for target
// ERR-BBSECTIONS: error: unsupported option '-fsplit-machine-functions=bbsections' for target

/// Disabling splitting is supported on every target.
// RUN: %clang -### --target=arm-unknown-linux -fsplit-machine-functions -fno-split-machine-functions %s
// RUN: %clang -### -c --target=arm-unknown-linux -fno-split-machine-functions %s
