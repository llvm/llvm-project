#if defined (SHARED) && IS_IN (libc)
# define STRNCMP __strncmp_ia32
# undef libc_hidden_builtin_def
# define libc_hidden_builtin_def(name)  \
    __hidden_ver1 (__strncmp_ia32, __GI_strncmp, __strncmp_ia32);
#endif

#include "string/strncmp.c"
