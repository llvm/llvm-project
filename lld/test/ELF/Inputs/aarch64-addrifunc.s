        .text
        .globl func1
        .type func1, %function
func1:
        adrp x8, :got: ifunc2
        ldr x8, [x8, :got_lo12: ifunc2]
        ret
