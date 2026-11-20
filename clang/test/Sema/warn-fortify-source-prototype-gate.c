// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x c %s -verify -Werror
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x c++ %s -verify -Werror
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x c %s -DWRONG_ARITY -verify -Werror
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x c++ %s -DWRONG_ARITY -verify -Werror
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x c %s -DWRONG_BUFFER -verify -Werror
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x c %s -DWRONG_COUNT -verify -Werror
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x c %s -DNO_BUILTIN -fno-builtin -verify -Werror
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x c++ %s -DNO_BUILTIN -fno-builtin -verify -Werror
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x c %s -DWRONG_FD -verify -Werror
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x c %s -DWRONG_OFFSET -verify -Werror
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x c %s -DWRONG_PATH -verify -Werror
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x c %s -DWRONG_CHAR_BUFFER -verify=expected,c -Werror
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x c++ %s -DWRONG_CHAR_BUFFER -verify -Werror
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x c %s -DWRONG_CHAR_PATH -verify -Werror
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c99 %s -DUNDECLARED -verify
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x c++ %s -DUNDECLARED -verify

// C exercises each argument guard. C++ additionally covers linkage and the
// differing getcwd redeclaration diagnostics; do not repeat the whole matrix.
#if !defined(UNDECLARED) && \
    (defined(__cplusplus) || defined(NO_BUILTIN) || \
     !(defined(WRONG_ARITY) || defined(WRONG_BUFFER) || defined(WRONG_COUNT) || \
       defined(WRONG_CHAR_BUFFER)))
// expected-no-diagnostics
#endif

#ifdef UNDECLARED
// Empty builtin prototypes must not provide implicit library declarations.
// getcwd has a complete prototype and provides the usual header hint in C.
void call_undeclared(void) {
  char buf[4];
  read(0, buf, 4); // expected-error {{undeclared}}
  write(0, buf, 4); // expected-error {{undeclared}}
  pread(0, buf, 4, 0); // expected-error {{undeclared}}
  pread64(0, buf, 4, 0); // expected-error {{undeclared}}
  pwrite(0, buf, 4, 0); // expected-error {{undeclared}}
  pwrite64(0, buf, 4, 0); // expected-error {{undeclared}}
  readlink("/", buf, 4); // expected-error {{undeclared}}
  readlinkat(0, "/", buf, 4); // expected-error {{undeclared}}
  getcwd(buf, 4); // expected-error {{undeclared}}
#ifndef __cplusplus
  // expected-note@-2 {{include the header <unistd.h> or explicitly provide a declaration for 'getcwd'}}
#endif
}
#else
// No __builtin_ aliases are provided. The library names are recognized
// unless builtin recognition is disabled. Ignore unrelated argument shapes,
// internal linkage, and C++ language linkage.
// getcwd has a complete builtin prototype, so incompatible C declarations
// receive the usual library redeclaration diagnostic.
typedef __SIZE_TYPE__ size_t;

#if __has_builtin(__builtin_read)
#error unexpected builtin alias
#endif
#ifndef NO_BUILTIN
#if !__has_builtin(read)
#error missing library builtin
#endif
#endif
#if __has_builtin(__builtin_write)
#error unexpected builtin alias
#endif
#ifndef NO_BUILTIN
#if !__has_builtin(write)
#error missing library builtin
#endif
#endif
#if __has_builtin(__builtin_pread)
#error unexpected builtin alias
#endif
#ifndef NO_BUILTIN
#if !__has_builtin(pread)
#error missing library builtin
#endif
#endif
#if __has_builtin(__builtin_pread64)
#error unexpected builtin alias
#endif
#ifndef NO_BUILTIN
#if !__has_builtin(pread64)
#error missing library builtin
#endif
#endif
#if __has_builtin(__builtin_pwrite)
#error unexpected builtin alias
#endif
#ifndef NO_BUILTIN
#if !__has_builtin(pwrite)
#error missing library builtin
#endif
#endif
#if __has_builtin(__builtin_pwrite64)
#error unexpected builtin alias
#endif
#ifndef NO_BUILTIN
#if !__has_builtin(pwrite64)
#error missing library builtin
#endif
#endif
#if __has_builtin(__builtin_readlink)
#error unexpected builtin alias
#endif
#ifndef NO_BUILTIN
#if !__has_builtin(readlink)
#error missing library builtin
#endif
#endif
#if __has_builtin(__builtin_readlinkat)
#error unexpected builtin alias
#endif
#ifndef NO_BUILTIN
#if !__has_builtin(readlinkat)
#error missing library builtin
#endif
#endif
#if __has_builtin(__builtin_getcwd)
#error unexpected builtin alias
#endif
#ifndef NO_BUILTIN
#if !__has_builtin(getcwd)
#error missing library builtin
#endif
#endif

#if defined(WRONG_CHAR_BUFFER) || defined(WRONG_CHAR_PATH)
// Check int and void pointees on separate functions so each call has only one
// invalid argument. A bad buffer must not conceal a missing path check.
#ifdef WRONG_CHAR_BUFFER
typedef const char *readlink_path_t;
typedef const char *readlinkat_path_t;
typedef int *readlink_buffer_t;
typedef void *readlinkat_buffer_t;
#else
typedef const void *readlink_path_t;
typedef const int *readlinkat_path_t;
typedef char *readlink_buffer_t;
typedef char *readlinkat_buffer_t;
#endif
#ifdef __cplusplus
extern "C" {
#endif
long readlink(readlink_path_t, readlink_buffer_t, size_t);
long readlinkat(int, readlinkat_path_t, readlinkat_buffer_t, size_t);
#ifdef WRONG_CHAR_BUFFER
char *getcwd(int *, size_t);
// c-error@-1 {{incompatible redeclaration of library function 'getcwd'}}
// c-note@-2 {{'getcwd' is a builtin with type 'char *(char *, __size_t)'}}
#endif
#ifdef __cplusplus
}
#endif

void call_unrelated_pointers(void) {
  char buf[4];
  readlink((readlink_path_t)0, (readlink_buffer_t)buf, 8);
  readlinkat(0, (readlinkat_path_t)0, (readlinkat_buffer_t)buf, 8);
#ifdef WRONG_CHAR_BUFFER
  getcwd((int *)buf, 8);
#endif
}
#elif defined(WRONG_FD) || defined(WRONG_OFFSET) || defined(WRONG_PATH)
#ifdef __cplusplus
extern "C" {
#endif
#ifdef WRONG_FD
int read(double, void *, size_t);
int write(double, const void *, size_t);
int pread(double, void *, size_t, long);
int pread64(double, void *, size_t, long long);
int pwrite(double, const void *, size_t, long);
int pwrite64(double, const void *, size_t, long long);
int readlinkat(double, const char *, char *, size_t);
#elif defined(WRONG_OFFSET)
int pread(int, void *, size_t, double);
int pread64(int, void *, size_t, double);
int pwrite(int, const void *, size_t, double);
int pwrite64(int, const void *, size_t, double);
#else
int readlink(int, char *, size_t);
int readlinkat(int, int, char *, size_t);
#endif
#ifdef __cplusplus
}
#endif

void call_mismatched_remaining_args(void) {
  char buf[4];
#ifdef WRONG_FD
  read(0, buf, 8);
  write(0, buf, 8);
  pread(0, buf, 8, 0);
  pread64(0, buf, 8, 0);
  pwrite(0, buf, 8, 0);
  pwrite64(0, buf, 8, 0);
  readlinkat(0, "/", buf, 8);
#elif defined(WRONG_OFFSET)
  pread(0, buf, 8, 0);
  pread64(0, buf, 8, 0);
  pwrite(0, buf, 8, 0);
  pwrite64(0, buf, 8, 0);
#else
  readlink(0, buf, 8);
  readlinkat(0, 0, buf, 8);
#endif
}
#elif defined(WRONG_ARITY) || defined(WRONG_BUFFER) || defined(WRONG_COUNT) || defined(NO_BUILTIN)
#ifdef WRONG_ARITY
#define LAST(x)
#else
#define LAST(x) , x
#endif
#ifdef WRONG_BUFFER
#define BUFFER int
#define BUF_ARG 0
#else
#define BUFFER void *
#define BUF_ARG buf
#endif
#ifdef WRONG_COUNT
#define COUNT double
#else
#define COUNT size_t
#endif

#ifdef __cplusplus
extern "C" {
#endif
int read(int, BUFFER LAST(COUNT));
int write(int, BUFFER LAST(COUNT));
int pread(int, BUFFER, COUNT LAST(long));
int pread64(int, BUFFER, COUNT LAST(long long));
int pwrite(int, BUFFER, COUNT LAST(long));
int pwrite64(int, BUFFER, COUNT LAST(long long));
int readlink(const char *, BUFFER LAST(COUNT));
int readlinkat(int, const char *, BUFFER LAST(COUNT));
int getcwd(BUFFER LAST(COUNT));
#if !defined(__cplusplus) && !defined(NO_BUILTIN)
// expected-error@-2 {{incompatible redeclaration of library function 'getcwd'}}
// expected-note@-3 {{'getcwd' is a builtin with type 'char *(char *, __size_t)'}}
#endif
#ifdef __cplusplus
}
#endif

void call_mismatched(void) {
  char buf[4];
  read(0, BUF_ARG LAST(8));
  write(0, BUF_ARG LAST(8));
  pread(0, BUF_ARG, 8 LAST(0));
  pread64(0, BUF_ARG, 8 LAST(0));
  pwrite(0, BUF_ARG, 8 LAST(0));
  pwrite64(0, BUF_ARG, 8 LAST(0));
  readlink("/", BUF_ARG LAST(8));
  readlinkat(0, "/", BUF_ARG LAST(8));
  getcwd(BUF_ARG LAST(8));
}
#else
static int read(int arg0, void * arg1, size_t arg2) { return 0; }
static int write(int arg0, const void * arg1, size_t arg2) { return 0; }
static int pread(int arg0, void * arg1, size_t arg2, long arg3) { return 0; }
static int pread64(int arg0, void * arg1, size_t arg2, long long arg3) { return 0; }
static int pwrite(int arg0, const void * arg1, size_t arg2, long arg3) { return 0; }
static int pwrite64(int arg0, const void * arg1, size_t arg2, long long arg3) { return 0; }
static int readlink(const char * arg0, char * arg1, size_t arg2) { return 0; }
static int readlinkat(int arg0, const char * arg1, char * arg2, size_t arg3) { return 0; }
static int getcwd(char * arg0, size_t arg1) { return 0; }
void call_static(void) {
  char buf[4];
  read(0, buf, 8);
  write(0, buf, 8);
  pread(0, buf, 8, 0);
  pread64(0, buf, 8, 0);
  pwrite(0, buf, 8, 0);
  pwrite64(0, buf, 8, 0);
  readlink("/", buf, 8);
  readlinkat(0, "/", buf, 8);
  getcwd(buf, 8);
}

#ifdef __cplusplus
namespace user {
int read(int, void *, size_t);
int write(int, const void *, size_t);
int pread(int, void *, size_t, long);
int pread64(int, void *, size_t, long long);
int pwrite(int, const void *, size_t, long);
int pwrite64(int, const void *, size_t, long long);
int readlink(const char *, char *, size_t);
int readlinkat(int, const char *, char *, size_t);
int getcwd(char *, size_t);
void call(void) {
  char buf[4];
  read(0, buf, 8);
  write(0, buf, 8);
  pread(0, buf, 8, 0);
  pread64(0, buf, 8, 0);
  pwrite(0, buf, 8, 0);
  pwrite64(0, buf, 8, 0);
  readlink("/", buf, 8);
  readlinkat(0, "/", buf, 8);
  getcwd(buf, 8);
}
} // namespace user
#endif
#endif
#endif // UNDECLARED
