.globl foo
.section .data.foo,"G",@,foo,comdat
foo:
        .int32 42
        .int32 43
        .size foo, 8
