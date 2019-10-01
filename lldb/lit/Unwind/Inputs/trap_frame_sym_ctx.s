        .text
        .globl  bar
bar:
        .cfi_startproc
        leal    (%edi, %edi), %eax
        ret
        .cfi_endproc

        .globl  asm_main
asm_main:
        .cfi_startproc
        pushq   %rbp
        .cfi_def_cfa_offset 16
        .cfi_offset %rbp, -16
        movq    %rsp, %rbp
        .cfi_def_cfa_register %rbp
        movl    $47, %edi

        # install tramp as return address
        # (similar to signal return trampolines on some platforms)
        leaq    tramp(%rip), %rax
        pushq   %rax
        jmp     bar # call, with return address pointing to tramp

        popq    %rbp
        .cfi_def_cfa %rsp, 8
        ret
        .cfi_endproc

        .globl  tramp
tramp:
        .cfi_startproc
        .cfi_signal_frame
        # Emit cfi to line up with the frame created by asm_main
        .cfi_def_cfa_offset 16
        .cfi_offset %rbp, -16
        .cfi_def_cfa_register %rbp
        # copy asm_main's epilog to clean up the frame
        popq    %rbp
        .cfi_def_cfa %rsp, 8
        ret
        .cfi_endproc
