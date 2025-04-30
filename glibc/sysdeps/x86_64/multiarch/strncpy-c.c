#define STRNCPY __strncpy_sse2
#undef libc_hidden_builtin_def
#define libc_hidden_builtin_def(strncpy)

#include <string/strncpy.c>
