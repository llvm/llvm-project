#define AS_STRNLEN
#define strlen __strnlen
#include "strlen.S"

weak_alias (__strnlen, strnlen);
libc_hidden_builtin_def (strnlen)
