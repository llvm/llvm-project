#include <dirent.h>
#include <glob.h>
#include <sys/stat.h>

#define glob_t glob64_t
#define globfree(pglob) globfree64 (pglob)

#include <posix/globfree.c>

libc_hidden_def (globfree64)
