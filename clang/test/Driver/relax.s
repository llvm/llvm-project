// REQUIRES: x86-registered-target
// RUN: %clang -### -c -integrated-as -Wa,--mrelax-relocations=yes %s 2>&1 | FileCheck  %s

// CHECK: "-cc1as"
// CHECK: "--mrelax-relocations"

// RUN: %clang -cc1as -triple x86_64-pc-linux --mrelax-relocations %s -o %t  -filetype obj
// RUN: llvm-readobj -r %t | FileCheck --check-prefix=REL %s

// REL: R_X86_64_REX_GOTPCRELX foo

        movq	foo@GOTPCREL(%rip), %rax
