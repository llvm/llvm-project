#define STRNLEN  __strnlen_ia32
#ifdef SHARED
# undef libc_hidden_def
# define libc_hidden_def(name)  \
    __hidden_ver1 (__strnlen_ia32, __GI_strnlen, __strnlen_ia32); \
    strong_alias (__strnlen_ia32, __strnlen_ia32_1); \
    __hidden_ver1 (__strnlen_ia32_1, __GI___strnlen, __strnlen_ia32_1);
#endif

#include "string/strnlen.c"
