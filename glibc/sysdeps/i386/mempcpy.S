#define USE_AS_MEMPCPY
#define MEMCPY		__mempcpy
#define MEMCPY_CHK	__mempcpy_chk
#include "memcpy.S"

weak_alias (__mempcpy, mempcpy)
libc_hidden_builtin_def (mempcpy)
