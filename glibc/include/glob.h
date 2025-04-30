#ifndef	_GLOB_H
#include <posix/glob.h>

#ifndef _ISOMAC
# include <sys/types.h>

libc_hidden_proto (glob)
libc_hidden_proto (glob64)
libc_hidden_proto (globfree)
libc_hidden_proto (globfree64)

# if __TIMESIZE == 64
#  define glob64_time64_t glob64_t
# else
# include <sys/stat.h>

typedef struct
  {
    size_t gl_pathc;
    char **gl_pathv;
    size_t gl_offs;
    int gl_flags;

    void (*gl_closedir) (void *);
    struct dirent64 *(*gl_readdir) (void *);
    void *(*gl_opendir) (const char *);
    int (*gl_lstat) (const char *__restrict, struct __stat64_t64 *__restrict);
    int (*gl_stat) (const char *__restrict, struct __stat64_t64 *__restrict);
  } glob64_time64_t;

extern int __glob64_time64 (const char *pattern, int flags,
			    int (*errfunc) (const char *, int),
			    glob64_time64_t *pglob);
libc_hidden_proto (__glob64_time64)
void __globfree64_time64 (glob64_time64_t *pglob);
libc_hidden_proto (__globfree64_time64)
# endif

/* Now define the internal interfaces.  */
extern int __glob_pattern_p (const char *__pattern, int __quote);
extern int __glob64 (const char *__pattern, int __flags,
		     int (*__errfunc) (const char *, int),
		     glob64_t *__pglob);
libc_hidden_proto (__glob64)
#endif

#endif
