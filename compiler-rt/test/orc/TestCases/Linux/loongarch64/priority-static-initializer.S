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
    .globl    destructor
    .p2align    2
    .type    destructor,@function
destructor:
.Ldestructor$local:

    pcalau12i    $a0, %pc_hi20(.L.str.2)
    addi.d    $a0, $a0, %pc_lo12(.L.str.2)
    b    %plt(puts)

    .globl    main
    .p2align    2
    .type    main,@function
main:

    addi.d    $sp, $sp, -16
    st.d    $ra, $sp, 8                     # 8-byte Folded Spill
    pcalau12i    $a0, %pc_hi20(.L.str)
    addi.d    $a0, $a0, %pc_lo12(.L.str)
    bl    %plt(puts)
    move    $a0, $zero
    ld.d    $ra, $sp, 8                     # 8-byte Folded Reload
    addi.d    $sp, $sp, 16
    ret

    .p2align    2
    .type    constructor.65535,@function
constructor.65535:

    addi.d    $sp, $sp, -16
    st.d    $ra, $sp, 8                     # 8-byte Folded Spill
    pcalau12i    $a0, %pc_hi20(.L.str.65535)
    addi.d    $a0, $a0, %pc_lo12(.L.str.65535)
    bl    %plt(puts)
    pcalau12i    $a0, %got_pc_hi20(__dso_handle)
    ld.d    $a0, $a0, %got_pc_lo12(__dso_handle)
    ld.d    $a2, $a0, 0
    pcalau12i    $a0, %pc_hi20(.Ldestructor$local)
    addi.d    $a0, $a0, %pc_lo12(.Ldestructor$local)
    move    $a1, $zero
    ld.d    $ra, $sp, 8                     # 8-byte Folded Reload
    addi.d    $sp, $sp, 16
    b    %plt(__cxa_atexit)

    .p2align    2
    .type    constructor.100,@function
constructor.100:

    addi.d    $sp, $sp, -16
    st.d    $ra, $sp, 8                     # 8-byte Folded Spill
    st.d    $fp, $sp, 0                     # 8-byte Folded Spill
    addi.d    $fp, $sp, 16
    pcalau12i    $a0, %pc_hi20(.L.str.100)
    addi.d    $a0, $a0, %pc_lo12(.L.str.100)
    bl    %plt(puts)
    ld.d    $fp, $sp, 0                     # 8-byte Folded Reload
    ld.d    $ra, $sp, 8                     # 8-byte Folded Reload
    addi.d    $sp, $sp, 16
    ret

    .p2align    2
    .type    constructor.200,@function
constructor.200:

    addi.d    $sp, $sp, -16
    st.d    $ra, $sp, 8                     # 8-byte Folded Spill
    st.d    $fp, $sp, 0                     # 8-byte Folded Spill
    addi.d    $fp, $sp, 16
    pcalau12i    $a0, %pc_hi20(.L.str.200)
    addi.d    $a0, $a0, %pc_lo12(.L.str.200)
    bl    %plt(puts)
    ld.d    $fp, $sp, 0                     # 8-byte Folded Reload
    ld.d    $ra, $sp, 8                     # 8-byte Folded Reload
    addi.d    $sp, $sp, 16
    ret

    .hidden    __dso_handle
    .type    .L.str,@object
    .section    .rodata.str1.1,"aMS",@progbits,1
.L.str:
    .asciz    "main"
    .size    .L.str, 5

    .type    .L.str.100,@object
.L.str.100:
    .asciz    "constructor 100"
    .size    .L.str.100, 16

    .type    .L.str.200,@object
.L.str.200:
    .asciz    "constructor 200"
    .size    .L.str.200, 16

    .type    .L.str.65535,@object
.L.str.65535:
    .asciz    "constructor 65535"
    .size    .L.str.65535, 18


    .type    .L.str.2,@object
.L.str.2:
    .asciz    "destructor"
    .size    .L.str.2, 11

    .section    .init_array.100,"aw",@init_array
    .p2align    3
    .dword    constructor.100
    .section    .init_array.200,"aw",@init_array
    .p2align    3
    .dword    constructor.200
    .section    .init_array,"aw",@init_array
    .p2align    3
    .dword    constructor.65535
