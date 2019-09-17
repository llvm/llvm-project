// An example of thunk generation that takes the maximum number of permitted
// passes to converge. We start with a set of branches of which all but one are
// in range. Any thunk added to extend the range of a branch is inserted in
// between the branches and the targets which knocks some more branches out
// of range. At the end of 9 passes of createThunks() every branch has a
// range extension thunk, allowing the final pass to check that no more thunks
// are required.
//
// As the size of the .text section changes 9 times, the symbol sym which
// depends on the size of .text will be updated 9 times. This test checks that
// any iteration limit to updating symbols does not limit thunk convergence.
// up to its pass limit without

        .thumb
        .section .text.00, "ax", %progbits
        .globl _start
        .thumb_func
_start: b.w f2
        b.w f2
        b.w f3
        b.w f3
        b.w f4
        b.w f4
        b.w f5
        b.w f5
        b.w f6
        b.w f6
        b.w f7
        b.w f7
        b.w f8
        b.w f8
        b.w f9
        b.w f9
        b.w f10
        b.w f10

        .section .text.01, "ax", %progbits
        .space 14 * 1024 * 1024
// Thunks are inserted here, initially only 1 branch is out of range and needs
// a thunk. However the added thunk is 4-bytes in size which makes another
// branch out of range, which adds another thunk ...
        .section .text.02, "ax", %progbits
        .space (2 * 1024 * 1024) - 68
        .thumb_func
f2:     bx lr
        nop
        .thumb_func
f3:     bx lr
        nop
        .thumb_func
f4:     bx lr
        nop
        .thumb_func
f5:     bx lr
        nop
        .thumb_func
f6:     bx lr
        nop
        .thumb_func
f7:     bx lr
        nop
        .thumb_func
f8:     bx lr
        nop
        .thumb_func
f9:     bx lr
        nop
        .thumb_func
f10:     bx lr
        nop
