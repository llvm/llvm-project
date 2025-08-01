// RUN: not %clang -### -c --target=x86_64 -Wa,--crel %s 2>&1 | FileCheck %s --check-prefix=NOEXP

// NOEXP: error: -Wa,--allow-experimental-crel must be specified to use -Wa,--crel. CREL is experimental and uses a non-standard section type code

// RUN: %clang -### -c --target=x86_64 -Wa,--crel,--allow-experimental-crel %s -Werror 2>&1 | FileCheck %s
// RUN: %clang -### -c --target=x86_64 -Wa,--crel,--no-crel,--allow-experimental-crel %s -Werror 2>&1 | FileCheck %s --check-prefix=NO
// RUN: %clang -### -c --target=x86_64 -Wa,--allow-experimental-crel %s -Werror 2>&1 | FileCheck %s --check-prefix=NO
// RUN: not %clang -### -c --target=arm64-apple-darwin -Wa,--crel,--allow-experimental-crel %s 2>&1 | FileCheck %s --check-prefix=ERR
// RUN: not %clang -### -c --target=mips64 -Wa,--crel,--allow-experimental-crel %s 2>&1 | FileCheck %s --check-prefix=ERR

// RUN: %clang -### -c --target=aarch64 -Werror -Wa,--crel,--allow-experimental-crel -x assembler %s -Werror 2>&1 | FileCheck %s --check-prefix=ASM
// RUN: not %clang -### -c --target=mips64 -Wa,--crel,--allow-experimental-crel -x assembler %s 2>&1 | FileCheck %s --check-prefix=ERR

// CHECK:  "-cc1" {{.*}}"--crel"
// NO:     "-cc1"
// NO-NOT: "--crel"
// ASM:    "-cc1as" {{.*}}"--crel"
// ERR: error: unsupported option '-Wa,--crel' for target '{{.*}}'

/// The --allow-experimental-crel error check is exempted for -fno-integrated-as.
// RUN: %clang -### -c --target=aarch64 -fno-integrated-as -Wa,--crel %s -Werror 2>&1 | FileCheck %s --check-prefix=GAS

// GAS: "--crel"

/// The --allow-experimental-crel error check doesn't apply to LTO.
// RUN: %clang -### --target=x86_64-linux -Werror -flto -Wa,--crel %s 2>&1 | FileCheck %s --check-prefix=LTO
// RUN: %clang -### --target=x86_64-linux -Werror -flto -Wa,--crel -Wa,--no-crel %s 2>&1 | FileCheck %s --check-prefix=LTO-NO

// LTO: "-plugin-opt=-crel"
// LTO-NO-NOT: "-plugin-opt=-crel"

// RUN: touch %t.o
// RUN: not %clang -### --target=mips64-linux-gnu -flto -Wa,--crel %t.o 2>&1 | FileCheck %s --check-prefix=ERR
