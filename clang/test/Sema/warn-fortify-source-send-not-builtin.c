// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x c %s -verify
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x c++ %s -verify
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x c %s -DMISMATCHED_SIG -verify
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x c++ %s -DMISMATCHED_SIG -verify
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x c %s -DMISMATCHED_SIG2 -verify
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x c++ %s -DMISMATCHED_SIG2 -verify
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x c %s -DMISMATCHED_SIG3 -verify
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x c++ %s -DMISMATCHED_SIG3 -verify
// expected-no-diagnostics

// Declarations that are not the POSIX send/sendto should not trigger
// -Wfortify-source diagnostics:
//   * a file-local send/sendto with internal linkage;
//   * in C++, a send/sendto without C language linkage;
//   * a C-linkage function whose argument count or parameter types do not match
//     POSIX send/sendto.

typedef unsigned long size_t;
typedef long ssize_t;
typedef unsigned int socklen_t;
struct sockaddr;
struct other_addr;

#ifdef MISMATCHED_SIG
#ifdef __cplusplus
extern "C" {
#endif
int send(int fd, const void *buf, size_t len);
int sendto(int fd, const void *buf, size_t len, int flags);
#ifdef __cplusplus
}
#endif

void call_mismatched_send(int fd) {
  char buf[10];
  (void)send(fd, buf, 20);
  (void)sendto(fd, buf, 20, 0);
}
#elif defined(MISMATCHED_SIG2)
#ifdef __cplusplus
extern "C" {
#endif
int send(const void *fd, const void *buf, size_t len, int flags);
int sendto(int fd, const void *buf, size_t len, int flags,
           const struct other_addr *addr, socklen_t addrlen);
#ifdef __cplusplus
}
#endif

void call_mismatched_send2(void) {
  char buf[10];
  (void)send(buf, buf, 20, 0);
  (void)sendto(0, buf, 20, 0, (const struct other_addr *)0, 0);
}
#elif defined(MISMATCHED_SIG3)
#ifdef __cplusplus
extern "C" {
#endif
int send(int fd, const void *buf, size_t len, const void *flags);
int sendto(int fd, const void *buf, size_t len, int flags,
           const struct sockaddr *addr, const socklen_t *addrlen);
#ifdef __cplusplus
}
#endif

void call_mismatched_send3(int fd) {
  char buf[10];
  (void)send(fd, buf, 20, (const void *)0);
  (void)sendto(fd, buf, 20, 0, (const struct sockaddr *)0, (const socklen_t *)0);
}
#else
static ssize_t send(int fd, const void *buf, size_t len, int flags) {
  (void)fd;
  (void)buf;
  (void)len;
  (void)flags;
  return 0;
}

static ssize_t sendto(int fd, const void *buf, size_t len, int flags,
                      const struct sockaddr *addr, socklen_t addrlen) {
  (void)fd;
  (void)buf;
  (void)len;
  (void)flags;
  (void)addr;
  (void)addrlen;
  return 0;
}

void call_static_send(int fd) {
  char buf[10];
  (void)send(fd, buf, 20, 0);
  (void)sendto(fd, buf, 20, 0, (const struct sockaddr *)0, 0);
}

#ifdef __cplusplus
namespace user {
ssize_t send(int, const void *, size_t, int);
ssize_t sendto(int, const void *, size_t, int, const struct sockaddr *,
               socklen_t);

void call(int fd) {
  char buf[10];
  (void)user::send(fd, buf, 20, 0);
  (void)user::sendto(fd, buf, 20, 0, nullptr, 0);
}
} // namespace user
#endif
#endif
