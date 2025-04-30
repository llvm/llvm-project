#include <stdint.h>

extern uintptr_t __stack_chk_guard;
#define STACK_CHK_GUARD __stack_chk_guard

#define POINTER_CHK_GUARD THREAD_GET_POINTER_GUARD()
