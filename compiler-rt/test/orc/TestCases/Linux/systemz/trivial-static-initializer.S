// Test that basic ELF static initializers work. The main function in this
// test returns the value of 'x', which is initially 1 in the data section,
// and reset to 0 if the _static_init function is run. If the static initializer
// does not run then main will return 1, causing the test to be treated as a
// failure.
//
// RUN: %clang -c -o %t %s
// RUN: %llvm_jitlink %t

	.text

        .globl  main         
        .p2align        4
main:
        lgrl    %r1, x@GOT
        lgf     %r2, 0(%r1)
        br      %r14

# static initializer sets the value of 'x' to zero.
        .p2align        4
_static_init:
        lgrl    %r1, x@GOT
        mvhi    0(%r1), 0
	br 	%r14

        .data
        .globl  x
        .p2align        2, 0x0
x:
        .long   1
        .size   x, 4

        .section        .init_array,"aw",@init_array
        .p2align        3, 0x0
        .quad   _static_init
