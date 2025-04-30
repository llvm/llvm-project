#define ffsl __something_else
#include <sysdeps/x86_64/ffs.c>
#undef ffsl
weak_alias (__ffs, ffsl)
