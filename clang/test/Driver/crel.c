// RUN: not %clang -### -c --target=x86_64 -Wa,--crel %s 2>&1 | FileCheck %s --check-prefix=NOEXP

// NOEXP: error: -Wa,--experimental-crel must be specified to use -Wa,--crel. CREL is experimental and takes a non-standard section type code

// RUN: %clang -### -c --target=x86_64 -Wa,--crel,--experimental-crel %s 2>&1 | FileCheck %s
// RUN: %clang -### -c --target=x86_64 -Wa,--crel,--no-crel,--experimental-crel %s 2>&1 | FileCheck %s --check-prefix=NO
// RUN: not %clang -### -c --target=arm64-apple-darwin -Wa,--crel,--experimental-crel %s 2>&1 | FileCheck %s --check-prefix=ERR
// RUN: not %clang -### -c --target=mips64 -Wa,--crel,--experimental-crel %s 2>&1 | FileCheck %s --check-prefix=ERR

// RUN: %clang -### -c --target=aarch64 -Werror -Wa,--crel,--experimental-crel -x assembler %s -Werror 2>&1 | FileCheck %s --check-prefix=ASM
// RUN: not %clang -### -c --target=mips64 -Wa,--crel,--experimental-crel -x assembler %s 2>&1 | FileCheck %s --check-prefix=ERR

// CHECK: "-cc1" {{.*}}"--crel"
// NO:     "-cc1"
// NO-NOT: "--crel"
// ASM:   "-cc1as" {{.*}}"--crel"
// ERR: error: unsupported option '-Wa,--crel' for target '{{.*}}'

/// Don't bother with --experimental-crel for LTO.
// RUN: %clang -### --target=x86_64-linux -Werror -flto -Wa,--crel %s 2>&1 | FileCheck %s --check-prefix=LTO
// LTO:       "-plugin-opt=-crel"

// RUN: touch %t.o
// RUN: not %clang -### --target=mips64-linux-gnu -flto -Wa,--crel %t.o 2>&1 | FileCheck %s --check-prefix=ERR
