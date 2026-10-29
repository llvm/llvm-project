// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x c %s -verify
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x c++ %s -verify
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x c %s -DMISMATCHED_SIG -verify
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x c++ %s -DMISMATCHED_SIG -verify
// expected-no-diagnostics

// Declarations that are not the POSIX recv/recvfrom should not trigger
// -Wfortify-source diagnostics:
//   * a file-local recv/recvfrom with internal linkage;
//   * in C++, a recv/recvfrom without C language linkage;
//   * a C-linkage function whose argument count or parameter types do not match
//     POSIX recv/recvfrom.

typedef unsigned long size_t;
typedef long ssize_t;
typedef unsigned int socklen_t;
struct sockaddr;

#ifdef MISMATCHED_SIG
#ifdef __cplusplus
extern "C" {
#endif
int recv(int fd, void *buf, size_t len);
int recvfrom(int fd, void *buf, size_t len, int flags);
#ifdef __cplusplus
}
#endif

void call_mismatched_recv(int fd) {
  char buf[10];
  (void)recv(fd, buf, 20);
  (void)recvfrom(fd, buf, 20, 0);
}
#else
static ssize_t recv(int fd, void *buf, size_t len, int flags) {
  (void)fd;
  (void)buf;
  (void)len;
  (void)flags;
  return 0;
}

static ssize_t recvfrom(int fd, void *buf, size_t len, int flags,
                        struct sockaddr *addr, socklen_t *addrlen) {
  (void)fd;
  (void)buf;
  (void)len;
  (void)flags;
  (void)addr;
  (void)addrlen;
  return 0;
}

void call_static_recv(int fd) {
  char buf[10];
  (void)recv(fd, buf, 20, 0);
  (void)recvfrom(fd, buf, 20, 0, (struct sockaddr *)0, (socklen_t *)0);
}

#ifdef __cplusplus
namespace user {
ssize_t recv(int, void *, size_t, int);
ssize_t recvfrom(int, void *, size_t, int, struct sockaddr *, socklen_t *);

void call(int fd) {
  char buf[10];
  (void)recv(fd, buf, 20, 0);
  (void)recvfrom(fd, buf, 20, 0, nullptr, nullptr);
}
} // namespace user
#endif
#endif
