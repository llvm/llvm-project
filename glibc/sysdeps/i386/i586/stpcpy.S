#define USE_AS_STPCPY
#define STRCPY __stpcpy

#include <sysdeps/i386/i586/strcpy.S>

weak_alias (__stpcpy, stpcpy)
libc_hidden_def (__stpcpy)
libc_hidden_builtin_def (stpcpy)
