        .text
        .globl  bar
        .type   bar, @function
bar:
.LFB0:
        .cfi_startproc
        leal    (%edi, %edi), %eax
        ret
        .cfi_endproc
.LFE0:
        .size   bar, .-bar
        .globl  foo
        .type   foo, @function
foo:
.LFB1:
        .cfi_startproc
        .cfi_escape 0x16, 0x10, 0x06, 0x38, 0x1c, 0x06, 0x08, 0x47, 0x1c
        call    bar
        addl    $1, %eax
        popq    %rdi
        subq    $0x47, %rdi
        jmp     *%rdi # Return
        .cfi_endproc
.LFE1:
        .size   foo, .-foo
        .globl  main
        .type   main, @function
main:
.LFB2:
        .cfi_startproc
        pushq   %rbp
        .cfi_def_cfa_offset 16
        .cfi_offset 6, -16
        movq    %rsp, %rbp
        .cfi_def_cfa_register 6
        movl    $47, %edi

        # Non-standard calling convention. The real return address must be
        # decremented by 0x47.
        leaq    0x47+1f(%rip), %rax
        pushq   %rax
        jmp     foo # call
1:
        popq    %rbp
        .cfi_def_cfa 7, 8
        ret
        .cfi_endproc
.LFE2:
        .size   main, .-main
