#define STRNCPY __strncpy_ia32
#ifdef SHARED
# undef libc_hidden_builtin_def
# define libc_hidden_builtin_def(name)  \
    __hidden_ver1 (__strncpy_ia32, __GI_strncpy, __strncpy_ia32);
#endif

#include "string/strncpy.c"
