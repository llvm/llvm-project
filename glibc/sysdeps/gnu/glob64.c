#include <dirent.h>
#include <glob.h>
#include <sys/stat.h>

#define dirent dirent64
#define __readdir(dirp) __readdir64 (dirp)

#define glob_t glob64_t
#define __glob __glob64
#define globfree(pglob) globfree64 (pglob)

#undef stat
#define stat stat64
#undef __stat
#define __stat(file, buf) __stat64 (file, buf)

#define COMPILE_GLOB64	1

#include <posix/glob.c>

libc_hidden_def (__glob64)
versioned_symbol (libc, __glob64, glob64, GLIBC_2_27);
libc_hidden_ver (__glob64, glob64)
