/* For __ARM_NEON__ this file defines memcpy.  */
#ifndef __ARM_NEON__
# define memcpy __memcpy_neon
# undef libc_hidden_builtin_def
# define libc_hidden_builtin_def(a)
#endif

#define MEMCPY_NEON
#include "memcpy_impl.S"
