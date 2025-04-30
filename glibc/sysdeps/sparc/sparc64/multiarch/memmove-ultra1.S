#define memmove __memmove_ultra1
#undef libc_hidden_builtin_def
#define libc_hidden_builtin_def(name)
#include <sysdeps/sparc/sparc64/memmove.S>
