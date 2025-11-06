## This test checks that trampolines are inserted in split fragments if
## necessary. There are 4 LSDA ranges with a landing pad to three landing pads.
## After splitting all blocks, there have to be 4 trampolines in the output.

# RUN: llvm-mc --filetype=obj --triple x86_64-unknown-unknown %s -o %t.o
# RUN: %clangxx %cxxflags %t.o -o %t.exe -Wl,-q -pie
# RUN: llvm-bolt %t.exe --split-functions --split-strategy=all --split-eh \
# RUN:         -o %t.bolt --print-split --print-only=main 2>&1 | FileCheck %s

# CHECK: -------   HOT-COLD SPLIT POINT   -------
# CHECK: .LFT0
# CHECK: Landing Pads: .LBB0
# CHECK: .LBB0
# CHECK-NEXT: Landing Pad
# CHECK: -------   HOT-COLD SPLIT POINT   -------
# CHECK: .Ltmp0
# CHECK: Landing Pads: .LBB1, .LBB2
# CHECK: .LBB1
# CHECK-NEXT: Landing Pad
# CHECK: .LBB2
# CHECK-NEXT: Landing Pad
# CHECK: -------   HOT-COLD SPLIT POINT   -------
# CHECK: -------   HOT-COLD SPLIT POINT   -------
# CHECK: -------   HOT-COLD SPLIT POINT   -------
# CHECK: -------   HOT-COLD SPLIT POINT   -------
# CHECK: -------   HOT-COLD SPLIT POINT   -------
# CHECK: .Ltmp3
# CHECK: Landing Pads: .LBB3
# CHECK: .LBB3
# CHECK-NEXT: Landing Pad
# CHECK: -------   HOT-COLD SPLIT POINT   -------
# CHECK: -------   HOT-COLD SPLIT POINT   -------

        .text
        .section        .rodata.str1.1,"aMS",@progbits,1
.LC0:
        .string "E"
.LC1:
        .string "C"
        .text
        .globl  main
        .type   main, @function
main:
.LFB1265:
        .cfi_startproc
        .cfi_personality 0x9b,DW.ref.__gxx_personality_v0
        .cfi_lsda 0x1b,.LLSDA1265
        pushq   %r12
        .cfi_def_cfa_offset 16
        .cfi_offset 12, -16
        pushq   %rbp
        .cfi_def_cfa_offset 24
        .cfi_offset 6, -24
        pushq   %rbx
        .cfi_def_cfa_offset 32
        .cfi_offset 3, -32
        testb   $3, %dil
        jne     .L13
        leaq    .LC1(%rip), %rdi
.LEHB0:
        call    puts@PLT
        # Trampoline to .L9
.LEHE0:
        jmp     .L11
.L13:
        movl    $16, %edi
        call    __cxa_allocate_exception@PLT
        movq    %rax, %rbx
        leaq    .LC0(%rip), %rsi
        movq    %rax, %rdi
.LEHB1:
        call    _ZNSt13runtime_errorC1EPKc@PLT
        # Trampoline to .L8
.LEHE1:
        movq    _ZNSt13runtime_errorD1Ev@GOTPCREL(%rip), %rdx
        movq    _ZTISt13runtime_error@GOTPCREL(%rip), %rsi
        movq    %rbx, %rdi
.LEHB2:
        call    __cxa_throw@PLT
        # Trampoline to .L9
.LEHE2:
.L9:
        movq    %rax, %rdi
        movq    %rdx, %rax
        jmp     .L4
.L8:
        movq    %rax, %r12
        movq    %rdx, %rbp
        movq    %rbx, %rdi
        call    __cxa_free_exception@PLT
        movq    %r12, %rdi
        movq    %rbp, %rax
.L4:
        cmpq    $1, %rax
        je      .L5
.LEHB3:
        call    _Unwind_Resume@PLT
.LEHE3:
.L5:
        call    __cxa_begin_catch@PLT
        movq    %rax, %rdi
        movq    (%rax), %rax
        call    *16(%rax)
        movq    %rax, %rdi
.LEHB4:
        call    puts@PLT
        # Trampoline to .L10
.LEHE4:
        call    __cxa_end_catch@PLT
.L11:
        movl    $0, %eax
        popq    %rbx
        .cfi_remember_state
        .cfi_def_cfa_offset 24
        popq    %rbp
        .cfi_def_cfa_offset 16
        popq    %r12
        .cfi_def_cfa_offset 8
        ret
.L10:
        .cfi_restore_state
        movq    %rax, %rbx
        call    __cxa_end_catch@PLT
        movq    %rbx, %rdi
.LEHB5:
        call    _Unwind_Resume@PLT
.LEHE5:
        .cfi_endproc
        .globl  __gxx_personality_v0
        .section        .gcc_except_table,"a",@progbits
        .align 4
.LLSDA1265:
        .byte   0xff
        .byte   0x9b
        .uleb128 .LLSDATT1265-.LLSDATTD1265
.LLSDATTD1265:
        .byte   0x1
        .uleb128 .LLSDACSE1265-.LLSDACSB1265
.LLSDACSB1265:
        .uleb128 .LEHB0-.LFB1265
        .uleb128 .LEHE0-.LEHB0
        .uleb128 .L9-.LFB1265
        .uleb128 0x1
        .uleb128 .LEHB1-.LFB1265
        .uleb128 .LEHE1-.LEHB1
        .uleb128 .L8-.LFB1265
        .uleb128 0x3
        .uleb128 .LEHB2-.LFB1265
        .uleb128 .LEHE2-.LEHB2
        .uleb128 .L9-.LFB1265
        .uleb128 0x1
        .uleb128 .LEHB3-.LFB1265
        .uleb128 .LEHE3-.LEHB3
        .uleb128 0
        .uleb128 0
        .uleb128 .LEHB4-.LFB1265
        .uleb128 .LEHE4-.LEHB4
        .uleb128 .L10-.LFB1265
        .uleb128 0
        .uleb128 .LEHB5-.LFB1265
        .uleb128 .LEHE5-.LEHB5
        .uleb128 0
        .uleb128 0
.LLSDACSE1265:
        .byte   0x1
        .byte   0
        .byte   0
        .byte   0x7d
        .align 4
        .long   DW.ref._ZTISt13runtime_error-.
.LLSDATT1265:
        .text
        .size   main, .-main
        .hidden DW.ref._ZTISt13runtime_error
        .weak   DW.ref._ZTISt13runtime_error
        .section        .data.rel.local.DW.ref._ZTISt13runtime_error,"awG",@progbits,DW.ref._ZTISt13runtime_error,comdat
        .align 8
        .type   DW.ref._ZTISt13runtime_error, @object
        .size   DW.ref._ZTISt13runtime_error, 8
DW.ref._ZTISt13runtime_error:
        .quad   _ZTISt13runtime_error
        .hidden DW.ref.__gxx_personality_v0
        .weak   DW.ref.__gxx_personality_v0
        .section        .data.rel.local.DW.ref.__gxx_personality_v0,"awG",@progbits,DW.ref.__gxx_personality_v0,comdat
        .align 8
        .type   DW.ref.__gxx_personality_v0, @object
        .size   DW.ref.__gxx_personality_v0, 8
DW.ref.__gxx_personality_v0:
        .quad   __gxx_personality_v0
        .ident  "GCC: (Compiler-Explorer-Build-gcc--binutils-2.38) 12.1.0"
        .section        .note.GNU-stack,"",@progbits
