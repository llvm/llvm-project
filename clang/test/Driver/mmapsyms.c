/// Alternative mapping symbol scheme for AArch64.
// RUN: %clang -### -c --target=aarch64 -Wa,-mmapsyms=implicit %s -Werror 2>&1 | FileCheck %s
// RUN: %clang -### -c --target=aarch64_be -Wa,-mmapsyms=implicit %s -Werror 2>&1 | FileCheck %s
// RUN: %clang -### -c --target=aarch64 -Wa,-mmapsyms=implicit,-mmapsyms=default %s -Werror 2>&1 | FileCheck %s --check-prefix=NO
// RUN: not %clang -### -c --target=arm64-apple-darwin -Wa,-mmapsyms=implicit %s 2>&1 | FileCheck %s --check-prefix=ERR
// RUN: not %clang -### -c --target=x86_64 -Wa,-mmapsyms=implicit %s 2>&1 | FileCheck %s --check-prefix=ERR2

// RUN: %clang -### -c --target=aarch64 -Werror -Wa,-mmapsyms=implicit -x assembler %s -Werror 2>&1 | FileCheck %s --check-prefix=ASM
// RUN: not %clang -### -c --target=x86_64 -Wa,-mmapsyms=implicit -x assembler %s 2>&1 | FileCheck %s --check-prefix=ERR2

// CHECK:  "-cc1" {{.*}}"-mmapsyms=implicit"
// NO:     "-cc1"
// NO-NOT: "-mmapsyms=implicit"
// ASM:    "-cc1as" {{.*}}"-mmapsyms=implicit"
// ERR: error: unsupported option '-Wa,-mmapsyms=' for target 'arm64-apple-darwin'
// ERR2: error: unsupported argument '-mmapsyms=implicit' to option '-Wa,'

/// Check LTO.
// RUN: %clang -### --target=aarch64-linux -Werror -flto -Wa,-mmapsyms=implicit %s 2>&1 | FileCheck %s --check-prefix=LTO
// RUN: %clang -### --target=aarch64-linux -Werror -flto -Wa,-mmapsyms=implicit -Wa,-mmapsyms=default %s 2>&1 | FileCheck %s --check-prefix=LTO-NO

// LTO: "-plugin-opt=-implicit-mapsyms"
// LTO-NO-NOT: "-plugin-opt=-implicit-mapsyms"

// RUN: touch %t.o
// RUN: not %clang -### --target=x86_64-unknown-linux -flto -Wa,-mmapsyms=implicit %t.o 2>&1 | FileCheck %s --check-prefix=LTO-ERR

// LTO-ERR: error: unsupported option '-Wa,-mmapsyms=' for target 'x86_64-unknown-linux'
