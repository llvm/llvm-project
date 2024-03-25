// RUN: %clang -### --target=x86_64 -fprofile-use=default.profdata -fsplit-machine-functions %s 2>&1 | FileCheck %s --check-prefixes=CHECK,LTO-NEG
// RUN: %clang -### --target=aarch64 -fprofile-use=default.profdata -fsplit-machine-functions %s 2>&1 | FileCheck %s --check-prefixes=CHECK,LTO-NEG
// RUN: %clang -### --target=x86_64 -fprofile-use=default.profdata -fsplit-machine-functions -fno-split-machine-functions %s -c 2>&1 | FileCheck -check-prefix=NEG %s

// CHECK:      "-fsplit-machine-functions"
// CHECK-SAME: "-fprofile-instrument-use-path=default.profdata"

// NEG-NOT:    "-fsplit-machine-functions"

// RUN: %clang -### --target=x86_64-linux -flto -fsplit-machine-functions %s 2>&1 | FileCheck %s --check-prefix=LTO
// RUN: %clang -### --target=x86_64-linux -flto -fsplit-machine-functions -fno-split-machine-functions %s 2>&1 | FileCheck %s --check-prefix=LTO-NEG

// LTO:         "-plugin-opt=-split-machine-functions"
// LTO-NEG-NOT: "-plugin-opt=-split-machine-functions"

// RUN: not %clang -### -c --target=arm-unknown-linux -fsplit-machine-functions %s 2>&1 | FileCheck %s --check-prefix=ERR
// ERR: error: unsupported option '-fsplit-machine-functions' for target

// RUN: %clang -### --target=arm-unknown-linux -fsplit-machine-functions -fno-split-machine-functions %s
