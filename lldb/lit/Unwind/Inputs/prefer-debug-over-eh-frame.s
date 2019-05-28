        .cfi_sections .eh_frame, .debug_frame
        .text
        .globl  bar
bar:
        .cfi_startproc
        leal    (%edi, %edi), %eax
        ret
        .cfi_endproc

        .globl  foo
foo:
        .cfi_startproc
        pushq   %rbp
        .cfi_def_cfa_offset 16
        .cfi_offset %rbp, -16
        movq    %rsp, %rbp
        .cfi_def_cfa_register %rbp
        call    bar
        addl    $1, %eax
        popq    %rbp
        ret
        .cfi_endproc

        .globl  asm_main
asm_main:
        .cfi_startproc
        pushq   %rbp
        .cfi_def_cfa_offset 16
        .cfi_offset 6, -16
        movq    %rsp, %rbp
        .cfi_def_cfa_register 6
        movl    $47, %edi

        call foo
        popq    %rbp
        .cfi_def_cfa 7, 8
        ret
        .cfi_endproc
