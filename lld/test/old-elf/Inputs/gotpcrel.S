        .text
        .globl  main
        .align  16, 0x90
        .type   main,@function
main:                                   # @main
        movq  blah@GOTPCREL(%rip), %rax
        movq  main@GOTPCREL(%rip), %rax
        ret

        .weak blah
        .type blah,@function
