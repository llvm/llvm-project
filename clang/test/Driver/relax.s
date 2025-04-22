// REQUIRES: x86-registered-target
// RUN: %clang -### -c --target=x86_64-pc-linux -integrated-as -Wa,--mrelax-relocations=no %s 2>&1 | FileCheck %s

// CHECK: "-cc1as"
// CHECK: "-mrelax-relocations=no"

// RUN: %clang -### -c --target=x86_64-pc-linux -integrated-as -Wa,--mapx-relax-relocations=yes %s 2>&1 | FileCheck %s --check-prefix=APXREL_OPTION

// APXREL_OPTION: "-cc1as"
// APXREL_OPTION: "-mapx-relax-relocations=yes"

// RUN: %clang -cc1as -triple x86_64-pc-linux %s -o %t -filetype obj -mapx-relax-relocations=yes
// RUN: llvm-readobj -r %t | FileCheck --check-prefix=APXREL %s
// RUN: %clang -cc1as -triple x86_64-pc-linux %s -o %t -filetype obj
// RUN: llvm-readobj -r %t | FileCheck --check-prefix=NOAPXREL %s

// APXREL: R_X86_64_REX_GOTPCRELX foo
// APXREL: R_X86_64_CODE_4_GOTPCRELX foo
// NOAPXREL: R_X86_64_REX_GOTPCRELX foo
// NOAPXREL: R_X86_64_GOTPCREL foo

        movq	foo@GOTPCREL(%rip), %rax
        movq	foo@GOTPCREL(%rip), %r16
