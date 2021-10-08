# REQUIRES: target={{x86_64-.+-linux-gnu}}
# RUN: %{build}
# RUN: %{run}

// TODO: Investigate these failures
// XFAIL: asan, tsan, ubsan

// TODO: Investigate this failure
// XFAIL: 32bits-on-64bits

# TODO: Investigate this failure on GCC.
# XFAIL: gcc

# The following assembly is a translation of this code:
#
#   _Unwind_Reason_Code callback(int, _Unwind_Action, long unsigned int,
#                                _Unwind_Exception*, _Unwind_Context*, void*) {
#     return _Unwind_Reason_Code(0);
#   }
#
#   int main() {
#     asm(".cfi_remember_state\n\t");
#     _Unwind_Exception exc;
#     _Unwind_ForcedUnwind(&exc, callback, 0);
#     asm(".cfi_restore_state\n\t");
#   }
#
# When unwinding, the CFI parser will stop parsing opcodes after the current PC,
# so in this case the DW_CFA_restore_state opcode will never be processed and,
# if the library doesn't clean up properly, the store allocated by
# DW_CFA_remember_state will be leaked.
#
# This test will fail when linked with an asan-enabled libunwind if the
# remembered state is leaked.

    SIZEOF_UNWIND_EXCEPTION = 32

    .text
callback:
    xorl    %eax, %eax
    retq

    .globl    main                    # -- Begin function main
    .p2align    4, 0x90
    .type    main,@function
main:                                   # @main
    .cfi_startproc
    subq    $8, %rsp   # Adjust stack alignment
    subq    $SIZEOF_UNWIND_EXCEPTION, %rsp
    .cfi_def_cfa_offset 48
    .cfi_remember_state
    movq    %rsp, %rdi
    movabsq $callback, %rsi
    xorl    %edx, %edx
    callq    _Unwind_ForcedUnwind
    .cfi_restore_state
    xorl    %eax, %eax
    addq    $SIZEOF_UNWIND_EXCEPTION, %rsp
    addq    $8, %rsp   # Undo stack alignment adjustment
    .cfi_def_cfa_offset 8
    retq
.Lfunc_end1:
    .size    main, .Lfunc_end1-main
    .cfi_endproc
                                        # -- End function
