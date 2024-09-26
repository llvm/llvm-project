// REQUIRES: x86-registered-target
// RUN: %clang -### -c -target x86_64-pc-linux -integrated-as -Wa,--mrelax-relocations=no %s 2>&1 | FileCheck %s

// CHECK: "-cc1as"
// CHECK: "-mrelax-relocations=no"

// RUN: %clang -cc1as -triple x86_64-pc-linux %s -o %t -filetype obj
// RUN: llvm-readobj -r %t | FileCheck --check-prefix=REL %s

// REL: R_X86_64_REX_GOTPCRELX foo
// REL: R_X86_64_REX2_GOTPCRELX foo

        movq	foo@GOTPCREL(%rip), %rax
        movq	foo@GOTPCREL(%rip), %r16
