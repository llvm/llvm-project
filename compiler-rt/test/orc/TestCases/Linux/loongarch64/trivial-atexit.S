// Test that the runtime correctly interposes atexit.
//
// REQUIRES: disabled
//   This test is disabled until a proper atexit interpose can be implemented:
//   the current one assumes that atexit is defined in the dylib that calls it,
//   which is not true in general. See
//   https://github.com/llvm/llvm-project/issues/74641
//
// RUN: %clang -c -o %t %s
// RUN: %llvm_jitlink %t

    .text
// OnExit destructor resets the test result override to zero.
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
    pcalau12i    $a0, %pc_hi20(on_exit)
    addi.d    $a0, $a0, %pc_lo12(on_exit)
    bl    %plt(atexit)
    ori    $a0, $zero, 1
    bl    %plt(llvm_jitlink_setTestResultOverride)
    move    $a0, $zero
    ld.d    $ra, $sp, 8                     # 8-byte Folded Reload
    addi.d    $sp, $sp, 16
    ret
.Lfunc_end1:
    .size    main, .Lfunc_end1-main
