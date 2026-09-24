// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x c %s -verify
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x c++ %s -verify
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x c %s -DMISMATCHED_SIG -verify
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x c++ %s -DMISMATCHED_SIG -verify
// expected-no-diagnostics

// Declarations that are not the POSIX/Bionic poll/ppoll/ppoll64 should not
// trigger -Wfortify-source diagnostics:
//   * a file-local poll/ppoll/ppoll64 with internal linkage;
//   * in C++, a poll/ppoll/ppoll64 without C language linkage;
//   * a C-linkage function whose argument count or parameter types do not match
//     POSIX poll/ppoll/ppoll64.

struct pollfd {
  int fd;
  short events;
  short revents;
};
struct timespec;
typedef unsigned long nfds_t;
typedef unsigned long sigset_t;
typedef unsigned long sigset64_t;

#ifdef MISMATCHED_SIG
#ifdef __cplusplus
extern "C" {
#endif
int poll(struct pollfd *fds, nfds_t nfds);
int ppoll(void *fds, nfds_t nfds, const struct timespec *tmo_p,
          const sigset_t *sigmask);
int ppoll64(struct pollfd *fds, const void *nfds, const struct timespec *tmo_p,
            const sigset64_t *sigmask);
#ifdef __cplusplus
}
#endif

void call_mismatched_poll(void) {
  struct pollfd fds[2];
  char buf[4];
  (void)poll(fds, 5);
  (void)ppoll(buf, 10, (const struct timespec *)0, (const sigset_t *)0);
  (void)ppoll64(fds, buf, (const struct timespec *)0, (const sigset64_t *)0);
}
#else
static int poll(struct pollfd *fds, nfds_t nfds, int timeout) {
  (void)fds;
  (void)nfds;
  (void)timeout;
  return 0;
}

static int ppoll(struct pollfd *fds, nfds_t nfds, const struct timespec *tmo_p,
                 const sigset_t *sigmask) {
  (void)fds;
  (void)nfds;
  (void)tmo_p;
  (void)sigmask;
  return 0;
}

static int ppoll64(struct pollfd *fds, nfds_t nfds,
                   const struct timespec *tmo_p, const sigset64_t *sigmask) {
  (void)fds;
  (void)nfds;
  (void)tmo_p;
  (void)sigmask;
  return 0;
}

void call_static_poll(void) {
  struct pollfd fds[2];
  (void)poll(fds, 5, 0);
  (void)ppoll(fds, 5, (const struct timespec *)0, (const sigset_t *)0);
  (void)ppoll64(fds, 5, (const struct timespec *)0, (const sigset64_t *)0);
}

#ifdef __cplusplus
namespace user {
int poll(struct pollfd *, nfds_t, int);
int ppoll(struct pollfd *, nfds_t, const struct timespec *, const sigset_t *);
int ppoll64(struct pollfd *, nfds_t, const struct timespec *,
            const sigset64_t *);

void call(void) {
  struct pollfd fds[2];
  (void)user::poll(fds, 5, 0);
  (void)user::ppoll(fds, 5, nullptr, nullptr);
  (void)user::ppoll64(fds, 5, nullptr, nullptr);
}
} // namespace user
#endif
#endif
