#define __pthread_timedblock __pthread_timedblock_intr
#define MSG_OPTIONS MACH_RCV_INTERRUPT
#include "pt-timedblock.c"
