#if defined(__arm__) && defined(__linux__)

#include "sanitizer_common/sanitizer_asm.h"

ASM_HIDDEN(COMMON_INTERCEPTOR_SPILL_AREA)

.comm _ZN14__interception10real_vforkE,4,4
.globl ASM_WRAPPER_NAME(vfork)
ASM_TYPE_FUNCTION(ASM_WRAPPER_NAME(vfork))
ASM_WRAPPER_NAME(vfork):
        // Save LR in the off-stack spill area.
        push    {r4, lr}
        bl      COMMON_INTERCEPTOR_SPILL_AREA
        pop     {r4, lr}
        str     lr, [r0]

        // Call real vfork. This may return twice. User code that runs between the first and the second return
        // may clobber the stack frame of the interceptor; that's why it does not have a frame.
        ldr     r0, .LCPI0_0
#if defined(__ARM_PIC_DATA_STATIC_BASE)
        ldr     r0, [r9, r0]
        ldr     r0, [r0]
#else
.LPC0_0:
        ldr     r0, [pc, r0]
#endif
        mov     lr, pc
        bx      r0

        push    {r0, r4}
        cmp     r0, #0
        beq     .L_exit

        // r0 != 0 => parent process. Clear stack shadow.
        add     r0, sp, #8
        bl      COMMON_INTERCEPTOR_HANDLE_VFORK

.L_exit:
        // Restore LR.
        bl      COMMON_INTERCEPTOR_SPILL_AREA
        ldr     lr, [r0]
        pop     {r0, r4}

        mov     pc, lr

.LCPI0_0:
#if defined(__ARM_PIC_DATA_STATIC_BASE)
        .long   _ZN14__interception10real_vforkE(GOT)
#else
        .long   _ZN14__interception10real_vforkE - (.LPC0_0+8)
#endif

ASM_SIZE(vfork)

ASM_INTERCEPTOR_TRAMPOLINE(vfork)
ASM_TRAMPOLINE_ALIAS(vfork, vfork)

#endif
