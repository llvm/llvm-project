        .text
        .section .text.foo,"G",@,foo,comdat
        .globl foo
        .type foo,@function
foo:
        .functype foo () -> ()
        return
        end_function

        .section .debug_foo,"G",@,foo,comdat
        .int32 2
        .section .debug_foo,"G",@,duplicate,comdat
        .int64 234
