// RUN: not %clang -### -c --target=x86_64 -Wa,--gsframe %s -Werror 2>&1 | FileCheck %s --check-prefix=NOEXP
// NOEXP: error: -Wa,--allow-experimental-sframe must be specified to use -Wa,--gsframe. SFrames are experimental and may be removed at any time without warning

// RUN: %clang -### -c --target=x86_64 -Wa,--gsframe -Wa,--allow-experimental-sframe %s -Werror 2>&1 | FileCheck %s
// CHECK: "-cc1"
// CHECK-SAME: "--gsframe"

// RUN: %clang -### -c --target=x86_64 %s 2>&1 | FileCheck %s --check-prefix=NO-GSFRAME
// NO-GSFRAME-NOT: "--gsframe"

// RUN: %clang -### -c --target=x86_64 -Werror -Wa,--gsframe,--allow-experimental-sframe -x assembler %s -Werror 2>&1 | FileCheck %s --check-prefix=ASM
// ASM: "-cc1as"
// ASM-SAME: "--gsframe"

// When aarch64 is supported, this will switch to a positive test.
// RUN: not %clang -### -c --target=aarch64 -Wa,--gsframe %s 2>&1 | FileCheck %s --check-prefix=NOTARGETAARCH
// NOTARGETAARCH: error: unsupported option '-Wa,--gsframe' for target '{{.*}}'

// RUN: not %clang -### -c --target=mips64 -Wa,--gsframe %s 2>&1 | FileCheck %s --check-prefix=NOTARGETC
// NOTARGETC: error: unsupported option '-Wa,--gsframe' for target '{{.*}}'

// RUN: not %clang -### -c --target=mips64 -Wa,--gsframe -x assembler %s 2>&1 | FileCheck %s --check-prefix=NOTARGETASM
// NOTARGETASM: error: unsupported option '-Wa,--gsframe' for target '{{.*}}'

