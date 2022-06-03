# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown \
# RUN:   %s -o %t.o
# RUN: link_fdata %s %t.o %t.fdata
# RUN: %clang %cflags %t.o -o %t.exe -Wl,-q

# RUN: llvm-bolt %t.exe -data %t.fdata -print-finalized \
# RUN:    -tail-duplication=moderate -tail-duplication-minimum-offset 1 -o %t.out | FileCheck %s

# FDATA: 1 main f 1 main 19 0 10
# FDATA: 1 main f 1 main 11 0 13
# FDATA: 1 main 17 1 main 3c 0 10
# FDATA: 1 main 39 1 main 3c 0 10

# CHECK: tail duplication modified 1 ({{.*}}%) functions; duplicated 1 blocks ({{.*}} bytes) responsible for {{.*}} dynamic executions ({{.*}} of all block executions)
# CHECK: BB Layout   : .LBB00, .Ltmp0, .Ltail-dup0, .Ltmp1, .Ltmp2

# This is the C++ code fed to Clang
# int fib(int term) {
#   if (term <= 1)
#     return term;
#   return fib(term-1) + fib(term-2);
# }

    .text
    .globl main
    .type main, %function
    .size main, .Lend-main
main:
    push   %rbp
    mov    %rsp,%rbp
    sub    $0x10,%rsp
    mov    %edi,-0x8(%rbp)
    cmpl   $0x1,-0x8(%rbp)
    jg     .BB1
.BB0:
    mov    -0x8(%rbp),%eax
    mov    %eax,-0x4(%rbp)
    jmp   .BB2
.BB1:
    mov    -0x8(%rbp),%edi
    sub    $0x1,%edi
    call   main
    mov    %eax,-0xc(%rbp)
    mov    -0x8(%rbp),%edi
    sub    $0x2,%edi
    call   main
    mov    %eax,%ecx
    mov    -0xc(%rbp),%eax
    add    %ecx,%eax
    mov    %eax,-0x4(%rbp)
.BB2:
    mov    -0x4(%rbp),%eax
    add    $0x10,%rsp
    pop    %rbp
    retq
    nopl   0x0(%rax)
.Lend:
