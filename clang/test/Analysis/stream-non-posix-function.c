// RUN: %clang_analyze_cc1 -fno-builtin -analyzer-checker=core,unix.Stream -verify %s
// RUN: %clang_analyze_cc1 -fno-builtin -analyzer-checker=core,alpha.unix.SimpleStream -verify %s

// expected-no-diagnostics

typedef struct _FILE FILE;

// These functions are not standard C library functions.
FILE *tmpfile(const char *restrict path); // Real 'tmpfile' should have exactly 0 formal parameters.
FILE *fopen(const char *restrict path);   // Real 'fopen' should have exactly 2 formal parameters.
FILE *fdopen(int fd);                     // Real 'fdopen' should have exactly 2 formal parameters.

void test_fopen_non_posix(void) {
  FILE *fp = fopen("file"); // no-leak: This isn't the standard POSIX `fopen`, we don't know the semantics of this call.
}

void test_tmpfile_non_posix(void) {
  FILE *fp = tmpfile("file"); // no-leak: This isn't the standard POSIX `tmpfile`, we don't know the semantics of this call.
}

void test_fdopen_non_posix(int fd) {
  FILE *fp = fdopen(fd); // no-leak: This isn't the standard POSIX `fdopen`, we don't know the semantics of this call.
}
