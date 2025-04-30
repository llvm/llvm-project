#if IS_IN (libc)
# define __strcspn_sse2 __strcspn_ia32
# include <sysdeps/x86_64/multiarch/strcspn-c.c>
#endif
