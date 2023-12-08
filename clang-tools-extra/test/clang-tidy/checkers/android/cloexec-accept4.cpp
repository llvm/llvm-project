// RUN: %check_clang_tidy %s android-cloexec-accept4 %t

typedef int socklen_t;
struct sockaddr {};

#define SOCK_NONBLOCK 1
#define __O_CLOEXEC 3
#define SOCK_CLOEXEC __O_CLOEXEC
#define TEMP_FAILURE_RETRY(exp) \
  ({                            \
    int _rc;                    \
    do {                        \
      _rc = (exp);              \
    } while (_rc == -1);        \
  })
#define NULL 0

extern "C" int accept4(int sockfd, struct sockaddr *addr, socklen_t *addrlen, int flags);

void a() {
  accept4(0, NULL, NULL, SOCK_NONBLOCK);
  // CHECK-MESSAGES: :[[@LINE-1]]:39: warning: 'accept4' should use SOCK_CLOEXEC where possible [android-cloexec-accept4]
  // CHECK-FIXES: accept4(0, NULL, NULL, SOCK_NONBLOCK | SOCK_CLOEXEC);
  TEMP_FAILURE_RETRY(accept4(0, NULL, NULL, SOCK_NONBLOCK));
  // CHECK-MESSAGES: :[[@LINE-1]]:58: warning: 'accept4'
  // CHECK-FIXES: TEMP_FAILURE_RETRY(accept4(0, NULL, NULL, SOCK_NONBLOCK | SOCK_CLOEXEC));
}

void f() {
  accept4(0, NULL, NULL, 3);
  // CHECK-MESSAGES: :[[@LINE-1]]:27: warning: 'accept4'
  // CHECK-FIXES: accept4(0, NULL, NULL, 3 | SOCK_CLOEXEC);
  TEMP_FAILURE_RETRY(accept4(0, NULL, NULL, 3));
  // CHECK-MESSAGES: :[[@LINE-1]]:46: warning: 'accept4'
  // CHECK-FIXES: TEMP_FAILURE_RETRY(accept4(0, NULL, NULL, 3 | SOCK_CLOEXEC));

  int flag = SOCK_NONBLOCK;
  accept4(0, NULL, NULL, flag);
  TEMP_FAILURE_RETRY(accept4(0, NULL, NULL, flag));
}

namespace i {
int accept4(int sockfd, struct sockaddr *addr, socklen_t *addrlen, int flags);

void d() {
  accept4(0, NULL, NULL, SOCK_NONBLOCK);
  TEMP_FAILURE_RETRY(accept4(0, NULL, NULL, SOCK_NONBLOCK));
}

} // namespace i

void e() {
  accept4(0, NULL, NULL, SOCK_CLOEXEC);
  TEMP_FAILURE_RETRY(accept4(0, NULL, NULL, SOCK_CLOEXEC));
  accept4(0, NULL, NULL, SOCK_NONBLOCK | SOCK_CLOEXEC);
  TEMP_FAILURE_RETRY(accept4(0, NULL, NULL, SOCK_NONBLOCK | SOCK_CLOEXEC));
}

class G {
public:
  int accept4(int sockfd, struct sockaddr *addr, socklen_t *addrlen, int flags);
  void d() {
    accept4(0, NULL, NULL, SOCK_NONBLOCK);
    TEMP_FAILURE_RETRY(accept4(0, NULL, NULL, SOCK_NONBLOCK));
  }
};
