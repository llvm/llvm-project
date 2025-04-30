#include <pt-internal.h>
#define RETTYPE error_t
#define RETURN(val) return val
#define __pthread_block __pthread_block_intr
#define MSG_OPTIONS MACH_RCV_INTERRUPT
#include "pt-block.c"
