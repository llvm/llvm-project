// Test that ELF static initializers with different constructor priorities work
// and are executed in the proper order.
//
// RUN: %clang -c -o %t %s
// RUN: %llvm_jitlink %t | FileCheck %s

// CHECK: constructor 100
// CHECK-NEXT: constructor 200
// CHECK-NEXT: constructor 65535
// CHECK-NEXT: main
// CHECK-NEXT: destructor

        .text

        .globl  destructor
        .p2align        4, 0x90
        .type   destructor,@function
destructor:
.Ldestructor$local:
        larl    %r2, .Lstr.d
        jg     puts@PLT

        .globl  main
        .p2align        4, 0x90
        .type   main,@function
main:
.Lmain$local:
        stmg    %r14, %r15, 112(%r15)
        aghi    %r15, -160
        larl    %r2, .Lstr
        brasl   %r14, puts@PLT
        lghi    %r2, 0
        lmg     %r14, %r15, 272(%r15)
        br      %r14

        .p2align        4
        .type   constructor.65535,@function
constructor.65535:
        stmg    %r14, %r15, 112(%r15)
        aghi    %r15, -160
        larl    %r2, .Lstr.65535
        brasl   %r14, puts@PLT
        lgrl    %r4, __dso_handle
        lgrl    %r2, .Ldestructor$local@GOT
        lghi    %r3, 0
        lmg     %r14, %r15, 272(%r15)
        jg      __cxa_atexit@PLT

        .p2align        4
        .type   constructor.200,@function
constructor.200:
        larl    %r2, .Lstr.200
        jg     puts@PLT

        .p2align        4
        .type   constructor.100,@function
constructor.100:
        larl    %r2, .Lstr.100
        jg     puts@PLT

        .hidden __dso_handle
        .section        .init_array.101,"aw",@init_array
        .p2align        3, 0x0
        .quad   constructor.100
        .section        .init_array.200,"aw",@init_array
        .p2align        3, 0x0
        .quad   constructor.200
        .section        .init_array,"aw",@init_array
        .p2align        3, 0x0
        .quad   constructor.65535
        .type   .Lstr.d,@object
        .section        .rodata.str1.2,"aMS",@progbits,1
        .p2align        1, 0x0
.Lstr.d:
        .asciz  "destructor"
        .size   .Lstr.d, 11

        .type   .Lstr.100,@object
        .p2align        1, 0x0
.Lstr.100:
        .asciz  "constructor 100"
        .size   .Lstr.100, 16

        .type   .Lstr.200,@object
        .p2align        1, 0x0
.Lstr.200:
        .asciz  "constructor 200"
        .size   .Lstr.200, 16

        .type   .Lstr.65535,@object
        .p2align        1, 0x0
.Lstr.65535:
        .asciz  "constructor 65535"
        .size   .Lstr.65535, 18

        .type   .Lstr,@object
        .p2align        1, 0x0
.Lstr:
        .asciz  "main"
        .size   .Lstr, 5
