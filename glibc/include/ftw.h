#ifndef _FTW_H
#include <io/ftw.h>

#ifndef _ISOMAC
# if __TIMESIZE != 64
#  include <sys/stat.h>

typedef int (*__ftw64_time64_func_t) (const char *,
				      const struct __stat64_t64 *, int);
typedef int (*__nftw64_time64_func_t) (const char *,
				       const struct __stat64_t64 *, int,
				       struct FTW *);

extern int __ftw64_time64 (const char *, __ftw64_time64_func_t, int);
extern int __nftw64_time64 (const char *, __nftw64_time64_func_t, int, int);
# endif
#endif

#endif /* _FTW_H  */
