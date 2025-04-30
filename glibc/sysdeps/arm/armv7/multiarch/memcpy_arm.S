#define memcpy __memcpy_arm
#undef libc_hidden_builtin_def
#define libc_hidden_builtin_def(a)
#include "memcpy_impl.S"

/* These versions of memcpy are defined not to clobber any VFP or NEON
   registers so they must always call the ARM variant of the memcpy code.  */
strong_alias (__memcpy_arm, __aeabi_memcpy)
strong_alias (__memcpy_arm, __aeabi_memcpy4)
strong_alias (__memcpy_arm, __aeabi_memcpy8)
