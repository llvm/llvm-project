// RUN: %clang -### -c --target=x86_64 -Wa,--gsframe %s -Werror 2>&1 | FileCheck %s
// CHECK:  "-cc1" {{.*}}"--gsframe"

// RUN: %clang -### -c --target=x86_64 %s 2>&1 | FileCheck %s --check-prefix=NO
// NO:     "-cc1"

// RUN: %clang -### -c --target=x86_64 -Werror -Wa,--gsframe -x assembler %s -Werror 2>&1 | FileCheck %s --check-prefix=ASM
// ASM:    "-cc1as" {{.*}}"--gsframe"

// RUN: not %clang -### -c --target=mips64 -Wa,--gsframe %s 2>&1 | FileCheck %s --check-prefix=NOTARGETC
// NOTARGETC: error: unsupported option '--gsframe' for target '{{.*}}'

// RUN: not %clang -### -c --target=mips64 -Wa,--gsframe -x assembler %s 2>&1 | FileCheck %s --check-prefix=NOTARGETASM
// NOTARGETASM: error: unsupported option '--gsframe' for target '{{.*}}'

