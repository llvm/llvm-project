#define USE_AS_BZERO
#define memset __bzero
#include <sysdeps/i386/i686/memset.S>
weak_alias (__bzero, bzero)
