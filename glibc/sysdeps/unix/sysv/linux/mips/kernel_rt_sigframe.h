#include <stdint.h>

/* Linux kernel RT signal frame. */
typedef struct kernel_rt_sigframe
  {
    uint32_t rs_ass[4];
    uint32_t rs_code[2];
    siginfo_t rs_info;
    ucontext_t rs_uc;
    uint32_t rs_altcode[8] __attribute__ ((__aligned__ (1 << 7)));
  }
kernel_rt_sigframe_t;
