#define STPNCPY __stpncpy_sse2
#undef weak_alias
#define weak_alias(ignored1, ignored2)
#undef libc_hidden_def
#define libc_hidden_def(stpncpy)

#include <string/stpncpy.c>
