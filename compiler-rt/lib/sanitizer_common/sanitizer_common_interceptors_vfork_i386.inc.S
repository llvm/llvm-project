#if defined(__i386__) && defined(__linux__)

#include "sanitizer_common/sanitizer_asm.h"

.comm _ZN14__interception10real_vforkE,4,4
.globl ASM_WRAPPER_NAME(vfork)
ASM_TYPE_FUNCTION(ASM_WRAPPER_NAME(vfork))
ASM_WRAPPER_NAME(vfork):
        // Store return address in the spill area and tear down the stack frame.
        sub     $12, %esp
        call    COMMON_INTERCEPTOR_SPILL_AREA
        mov     12(%esp), %ecx
        mov     %ecx, (%eax)
        add     $16, %esp


        call    *_ZN14__interception10real_vforkE

        // Restore return address from the spill area.
        sub     $16, %esp
        mov     %eax, 4(%esp)
        call    COMMON_INTERCEPTOR_SPILL_AREA
        mov     (%eax), %ecx
        mov     %ecx, 12(%esp)
        mov     4(%esp), %eax


        // Call handle_vfork in the parent process (%rax != 0).
        test    %eax, %eax
        je      .L_exit

        lea     16(%esp), %ecx
        mov     %ecx, (%esp)
        call    COMMON_INTERCEPTOR_HANDLE_VFORK@PLT

.L_exit:
        mov     4(%esp), %eax
        add     $12, %esp
        ret
ASM_SIZE(vfork)

.weak vfork
.set vfork, ASM_WRAPPER_NAME(vfork)

#endif
