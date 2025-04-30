#undef libc_hidden_builtin_def
#define libc_hidden_builtin_def(name)
#define memmove __memmove_ultra1
#include <sysdeps/sparc/sparc32/sparcv9/memmove.S>
