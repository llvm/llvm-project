// RUN: %clang_cc1 -fsyntax-only %s
// Make sure OpenBSD's bounded extension is accepted.

typedef long ssize_t;
typedef unsigned long size_t;
typedef struct FILE FILE;

ssize_t read(int, void *, size_t)
    __attribute__((__bounded__(__buffer__,2,3)));
int readlink(const char *, char *, size_t)
    __attribute__((__bounded__(__string__,2,3)));
size_t fread(void *, size_t, size_t, FILE *)
    __attribute__((__bounded__(__size__,1,3,2)));
char *getwd(char *)
    __attribute__((__bounded__(__minbytes__,1,1024)));
