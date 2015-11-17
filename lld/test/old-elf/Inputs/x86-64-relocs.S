        .text
        .globl  main
        .align  16, 0x90
        .type   main,@function
main:                                   # @main
        call foo@PLT
        ret

        .globl foo
        .type foo,@function
foo:
        ret
