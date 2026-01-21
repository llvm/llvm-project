// RUN: %clang_cc1 -triple x86_64-apple-macosx10.14.0 %s -verify
// RUN: %clang_cc1 -triple x86_64-linux %s -verify
// RUN: %clang_cc1 -triple x86_64-linux %s -verify -DO_CREAT='(32 | __linux__)'


#define FAUX_CREATE 0100
#if O_CREAT != FAUX_CREATE
void call_open_no_creat(void) {
  __builtin_open("name", FAUX_CREATE, 0777);
  __builtin_open("name", FAUX_CREATE);
}
#endif

#define O_RDONLY 0
#define O_WRONLY 01
#define O_RDWR   02
#ifndef O_CREAT
#define O_CREAT 0100
#endif
#define __O_DIRECTORY  0x10000
#define __O_TMPFILE   (020000000 | __O_DIRECTORY)
#define O_TMPFILE  __O_TMPFILE /* Atomically create nameless file.  */

void call_open(void) {
#if O_CREAT == 64
  __builtin_open("name", 64); // expected-warning {{nonzero 'mode' argument must be specified as the flag 'O_CREAT' would result in file creation}}
#endif
  __builtin_open("name", O_TMPFILE | O_RDONLY); // expected-warning {{nonzero 'mode' argument must be specified as the flag 'O_TMPFILE' would result in file creation}}
  __builtin_open("name", O_TMPFILE + O_RDONLY); // expected-warning {{nonzero 'mode' argument must be specified as the flag 'O_TMPFILE' would result in file creation}}
  __builtin_open("name", O_TMPFILE); // expected-warning {{nonzero 'mode' argument must be specified as the flag 'O_TMPFILE' would result in file creation}}
  __builtin_open("name", O_CREAT); // expected-warning {{nonzero 'mode' argument must be specified as the flag 'O_CREAT' would result in file creation}}
  __builtin_open("name", O_CREAT | O_TMPFILE); // expected-warning {{nonzero 'mode' argument must be specified as the flags 'O_CREAT' and 'O_TMPFILE' would result in file creation}}
  __builtin_open("name", O_CREAT | O_TMPFILE, 0777);
  __builtin_open("name", O_CREAT | O_TMPFILE, 0777, 0); // expected-warning {{too many arguments passed to 'open'; it expects a maximum of 1 variadic parameter}}
  __builtin_open("name", O_CREAT | O_TMPFILE, 0777, 0, 0); // expected-warning {{too many arguments passed to 'open'; it expects a maximum of 1 variadic parameter}}
}

#ifdef __cplusplus
extern "C" {
#endif

typedef unsigned int uint32_t;
typedef uint32_t mode_t;

mode_t umode(mode_t);
int open(const char *pathname, int flags, ... /* mode_t mode */ );
int open64(const char *pathname, int flags, ... /* mode_t mode */ );
int openat(int fddir, const char *pathname, int flags, ... /* mode_t mode */ );
int openat64(int fddir, const char *pathname, int flags, ... /* mode_t mode */ );

#ifdef __cplusplus
}
#endif

void call_openat(void) {
  __builtin_openat(0, "name", O_CREAT, 0777);
  __builtin_openat(0, "name", O_CREAT, 01000);
#if !defined(__linux__)
  // expected-warning@-2{{invalid mode}}
#endif
}

void call_umask(void) {
  __builtin_umask(0);
  __builtin_umask(0777);
  __builtin_umask(01000); // expected-warning {{invalid mode}}
}

#if defined(__APPLE__)
#define PATH_MAX 1024
#elif defined(__linux__)
#define PATH_MAX 4096
#endif

void call_realpath() {
  char too_small[PATH_MAX - 1];
  char too_big[PATH_MAX + 1];
  char too_just_right[PATH_MAX];

  __builtin_realpath("hah", too_small); // expected-warning-re {{'realpath' distination buffer needs to be larger than than PATH_MAX bytes ({{[0-9]+}}), but buffer is {{[0-9]+}}}}
  __builtin_realpath("hah", too_big);
  __builtin_realpath("hah", too_just_right);
}

# 1 "poll.h" 1 3
# 1 "sys/poll.h" 1 3

#if defined(__APPLE__)
typedef unsigned int nfds_t;
#elif defined(__linux__)
typedef unsigned long int nfds_t;
#endif

struct pollfd {
  int fd;
  short events;
  short revents;
};
extern int poll (struct pollfd *__fds, nfds_t __nfds, int __timeout);

# 2 "poll_test.c" 2

#define __builtin_poll poll

void call_poll(void) {
  struct pollfd fds[] = {
    {0, 0, 0},
    {0, 0, 0},
    {0, 0, 0},
    {0, 0, 0},
    {0, 0, 0},
    {0, 0, 0},
    {0, 0, 0},
    {0, 0, 0},
    {0, 0, 0},
  };
  const nfds_t nfds = sizeof(fds) / sizeof(*fds);
  __builtin_poll(fds, nfds, 0);
  __builtin_poll(fds, nfds + 1, 0); // expected-warning {{the element count value '10' is higher than the number of elements in the array '9'}}
  __builtin_poll(fds, nfds - 1, 0);
  /* Unhandled errors */
  __builtin_poll(&fds[1], nfds, 0);
  __builtin_poll(fds + 1, nfds, 0);
  __builtin_poll(fds - 1, nfds, 0);
}
