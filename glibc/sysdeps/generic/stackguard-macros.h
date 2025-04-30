#include <stdint.h>

extern uintptr_t __stack_chk_guard;
#define STACK_CHK_GUARD __stack_chk_guard

#ifdef PTRGUARD_LOCAL
extern uintptr_t __pointer_chk_guard_local;
# define POINTER_CHK_GUARD __pointer_chk_guard_local
#else
extern uintptr_t __pointer_chk_guard;
# define POINTER_CHK_GUARD __pointer_chk_guard
#endif
