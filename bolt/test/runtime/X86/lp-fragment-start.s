# Test that landing pads at fragment entries are not omitted during unwinding.
# This uses profile2 splitting with a fake fdata, so SplitFunctions splits main
# right before landing pad.

# RUN: llvm-mc --filetype=obj --triple x86_64-unknown-unknown %s -o %t.o
# RUN: link_fdata %s %t.o %t.fdata
# RUN: llvm-strip --strip-unneeded %t.o
# RUN: %clang++ %cflags %t.o -o %t.exe -Wl,-q
# RUN: llvm-bolt %t.exe -o %t.bolt --split-functions --data=%t.fdata \
# RUN:         --print-split --print-only=main --split-eh --split-all-cold \
# RUN:     2>&1 | FileCheck --check-prefix=BOLT-CHECK %s
# RUN: %t.bolt | FileCheck --check-prefix=RUN-CHECK %s

# BOLT-CHECK: -------   HOT-COLD SPLIT POINT   -------
# BOLT-CHECK-EMPTY:
# BOLT-CHECK-NEXT: .LLP1

# RUN-CHECK: failed successfully


        .text
        .file   "lp-fragment-start.cpp"
        .globl  main                            # -- Begin function main
        .p2align        4, 0x90
        .type   main,@function
main:                                   # @main
.Lfunc_begin0:
        .cfi_startproc
        .cfi_personality 155, DW.ref.__gxx_personality_v0
        .cfi_lsda 27, .Lexception0
        pushq   %r15
        .cfi_def_cfa_offset 16
        pushq   %r14
        .cfi_def_cfa_offset 24
        pushq   %rbx
        .cfi_def_cfa_offset 32
        .cfi_offset %rbx, -32
        .cfi_offset %r14, -24
        .cfi_offset %r15, -16
        movl    $16, %edi
LL_fdata0:
        callq   __cxa_allocate_exception@PLT
# FDATA: 1 main #LL_fdata0# 1 __cxa_allocate_exception@PLT 0 0 1
        movq    %rax, %rbx
.Ltmp0:
        leaq    .L.str(%rip), %rsi
        movq    %rax, %rdi
        callq   _ZNSt13runtime_errorC1EPKc@PLT
        jmp     .Ltmp3
.Ltmp1:
# Cause split here, so .Ltmp5 is the first block of the fragment
.Ltmp5: # LP for .Ltmp3 to .Ltmp4
        movq    %rdx, %r15
        movq    %rax, %r14
        jmp     .LBB0_4
.Ltmp3: # throw std::runtime_error
        movq    _ZTISt13runtime_error@GOTPCREL(%rip), %rsi
        movq    _ZNSt13runtime_errorD1Ev@GOTPCREL(%rip), %rdx
        movq    %rbx, %rdi
        callq   __cxa_throw@PLT # LP: .Ltmp5
.Ltmp4:
.Ltmp2:
        movq    %rdx, %r15
        movq    %rax, %r14
        movq    %rbx, %rdi
        callq   __cxa_free_exception@PLT
.LBB0_4:
        movq    %r14, %rdi
        cmpl    $1, %r15d
        jne     .LBB0_5
        callq   __cxa_begin_catch@PLT
        movq    (%rax), %rcx
        movq    %rax, %rdi
        callq   *16(%rcx)
        movq    %rax, %rdi
        callq   puts@PLT
        callq   __cxa_end_catch@PLT
        xorl    %eax, %eax
        popq    %rbx
        .cfi_def_cfa_offset 24
        popq    %r14
        .cfi_def_cfa_offset 16
        popq    %r15
        .cfi_def_cfa_offset 8
        retq
.LBB0_5:
        .cfi_def_cfa_offset 32
        callq   _Unwind_Resume@PLT
.Lfunc_end0:
        .size   main, .Lfunc_end0-main
        .cfi_endproc
        .section        .gcc_except_table,"a",@progbits
        .p2align        2
GCC_except_table0:
.Lexception0:
        .byte   255                             # @LPStart Encoding = omit
        .byte   155                             # @TType Encoding = indirect pcrel sdata4
        .uleb128 .Lttbase0-.Lttbaseref0
.Lttbaseref0:
        .byte   1                               # Call site Encoding = uleb128
        .uleb128 .Lcst_end0-.Lcst_begin0
.Lcst_begin0:
        .uleb128 .Lfunc_begin0-.Lfunc_begin0    # >> Call Site 1 <<
        .uleb128 .Ltmp0-.Lfunc_begin0           #   Call between .Lfunc_begin0 and .Ltmp0
        .byte   0                               #     has no landing pad
        .byte   0                               #   On action: cleanup
        .uleb128 .Ltmp0-.Lfunc_begin0           # >> Call Site 2 <<
        .uleb128 .Ltmp1-.Ltmp0                  #   Call between .Ltmp0 and .Ltmp1
        .uleb128 .Ltmp2-.Lfunc_begin0           #     jumps to .Ltmp2
        .byte   3                               #   On action: 2
        .uleb128 .Ltmp3-.Lfunc_begin0           # >> Call Site 3 <<
        .uleb128 .Ltmp4-.Ltmp3                  #   Call between .Ltmp3 and .Ltmp4
        .uleb128 .Ltmp5-.Lfunc_begin0           #     jumps to .Ltmp5
        .byte   5                               #   On action: 3
        .uleb128 .Ltmp4-.Lfunc_begin0           # >> Call Site 4 <<
        .uleb128 .Lfunc_end0-.Ltmp4             #   Call between .Ltmp4 and .Lfunc_end0
        .byte   0                               #     has no landing pad
        .byte   0                               #   On action: cleanup
.Lcst_end0:
        .byte   0                               # >> Action Record 1 <<
        .byte   0                               #   No further actions
        .byte   1                               # >> Action Record 2 <<
        .byte   125                             #   Continue to action 1
        .byte   1                               # >> Action Record 3 <<
        .byte   0                               #   No further actions
        .p2align        2
.Ltmp6:                                 # TypeInfo 1
        .long   .L_ZTISt13runtime_error.DW.stub-.Ltmp6
.Lttbase0:
        .p2align        2
        .type   .L.str,@object                  # @.str
        .section        .rodata.str1.1,"aMS",@progbits,1
.L.str:
        .asciz  "failed successfully"
        .size   .L.str, 20

        .data
        .p2align        3
.L_ZTISt13runtime_error.DW.stub:
        .quad   _ZTISt13runtime_error
        .hidden DW.ref.__gxx_personality_v0
        .weak   DW.ref.__gxx_personality_v0
        .section        .data.DW.ref.__gxx_personality_v0,"aGw",@progbits,DW.ref.__gxx_personality_v0,comdat
        .p2align        3
        .type   DW.ref.__gxx_personality_v0,@object
        .size   DW.ref.__gxx_personality_v0, 8
DW.ref.__gxx_personality_v0:
        .quad   __gxx_personality_v0
        .ident  "clang version 14.0.0 (https://github.com/llvm/llvm-project.git 329fda39c507e8740978d10458451dcdb21563be)"
        .section        ".note.GNU-stack","",@progbits
        .addrsig
        .addrsig_sym __gxx_personality_v0
        .addrsig_sym _Unwind_Resume
        .addrsig_sym _ZTISt13runtime_error


# #include <cstdio>
# #include <stdexcept>
#
# int main() {
#     try {
#         throw std::runtime_error("failed successfully");
#     } catch (const std::runtime_error &e) {
#         puts(e.what());
#     }
#     return 0;
# }
