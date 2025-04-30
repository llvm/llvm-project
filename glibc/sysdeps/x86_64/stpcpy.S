#define USE_AS_STPCPY
#define STRCPY __stpcpy

#include <sysdeps/x86_64/strcpy.S>

weak_alias (__stpcpy, stpcpy)
libc_hidden_def (__stpcpy)
libc_hidden_builtin_def (stpcpy)
