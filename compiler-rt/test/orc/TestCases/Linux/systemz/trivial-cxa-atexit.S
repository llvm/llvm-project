// Test that the runtime correctly interposes ___cxa_atexit.
//
// RUN: %clang -c -o %t %s
// RUN: %llvm_jitlink %t

        .text
// OnExit destructor resets the test result override to zero.
        .section        .text._ZN6OnExitD2Ev,"axG",@progbits,_ZN6OnExitD2Ev,comdat
        .p2align        4
        .type   _ZN6OnExitD2Ev,@function
_ZN6OnExitD2Ev:
        .cfi_startproc
        lghi    %r2, 0
        jg llvm_jitlink_setTestResultOverride@PLT
        .cfi_endproc

// main registers the atexit and sets the test result to one.
        .globl  main
        .p2align        4
        .type   main,@function
main:
        .cfi_startproc
# %bb.0:
        stmg    %r14, %r15, 48(%r15)
        lgrl    %r2, _ZN6OnExitD2Ev@GOT
        lgrl    %r4, __dso_handle@GOT
        larl    %r3, _ZL6onExit
        brasl   %r14, __cxa_atexit@PLT
        lghi    %r2, 1
        brasl   %r14, llvm_jitlink_setTestResultOverride@PLT
        lghi    %r2, 0
        lmg     %r14, %r15, 48(%r15)
        br      %r14
.Lfunc_end1:
        .size   main, .Lfunc_end1-main
        .cfi_endproc

        .type   _ZL6onExit,@object
        .local  _ZL6onExit
        .comm   _ZL6onExit,1,2
        .hidden __dso_handle
