// Test that the runtime correctly interposes ___cxa_atexit.
//
// RUN: %clang -c -o %t %s
// RUN: %llvm_jitlink %t

    .text
// Destructor resets the test result override to zero.
    .globl    on_exit
    .p2align    2
    .type    on_exit,@function
on_exit:

    move    $a0, $zero
    b    %plt(llvm_jitlink_setTestResultOverride)
.Lfunc_end0:
    .size    on_exit, .Lfunc_end0-on_exit

// main registers the atexit and sets the test result to one.
    .globl    main
    .p2align    2
    .type    main,@function
main:

    addi.d    $sp, $sp, -16
    st.d    $ra, $sp, 8                     # 8-byte Folded Spill
    pcalau12i    $a0, %got_pc_hi20(__dso_handle)
    ld.d    $a0, $a0, %got_pc_lo12(__dso_handle)
    ld.d    $a2, $a0, 0
    pcalau12i    $a0, %pc_hi20(on_exit)
    addi.d    $a0, $a0, %pc_lo12(on_exit)
    move    $a1, $zero
    bl    %plt(__cxa_atexit)
    ori    $a0, $zero, 1
    bl    %plt(llvm_jitlink_setTestResultOverride)
    move    $a0, $zero
    ld.d    $ra, $sp, 8                     # 8-byte Folded Reload
    addi.d    $sp, $sp, 16
    ret
.Lfunc_end1:
    .size    main, .Lfunc_end1-main
