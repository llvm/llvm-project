// RUN: %check_clang_tidy %s android-cloexec-epoll-create1 %t

#define __O_CLOEXEC 3
#define EPOLL_CLOEXEC __O_CLOEXEC
#define TEMP_FAILURE_RETRY(exp) \
  ({                            \
    int _rc;                    \
    do {                        \
      _rc = (exp);              \
    } while (_rc == -1);        \
  })

extern "C" int epoll_create1(int flags);

void a() {
  epoll_create1(0);
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: 'epoll_create1' should use EPOLL_CLOEXEC where possible [android-cloexec-epoll-create1]
  // CHECK-FIXES: epoll_create1(EPOLL_CLOEXEC);
  TEMP_FAILURE_RETRY(epoll_create1(0));
  // CHECK-MESSAGES: :[[@LINE-1]]:37: warning: 'epoll_create1'
  // CHECK-FIXES: TEMP_FAILURE_RETRY(epoll_create1(EPOLL_CLOEXEC));
}

void f() {
  epoll_create1(3);
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: 'epoll_create1'
  // CHECK-FIXES: epoll_create1(EPOLL_CLOEXEC);
  TEMP_FAILURE_RETRY(epoll_create1(3));
  // CHECK-MESSAGES: :[[@LINE-1]]:37: warning: 'epoll_create1'
  // CHECK-FIXES: TEMP_FAILURE_RETRY(epoll_create1(EPOLL_CLOEXEC));

  int flag = 0;
  epoll_create1(EPOLL_CLOEXEC);
  TEMP_FAILURE_RETRY(epoll_create1(EPOLL_CLOEXEC));
}

namespace i {
int epoll_create1(int flags);

void d() {
  epoll_create1(0);
  TEMP_FAILURE_RETRY(epoll_create1(0));
}

} // namespace i

void e() {
  epoll_create1(EPOLL_CLOEXEC);
  TEMP_FAILURE_RETRY(epoll_create1(EPOLL_CLOEXEC));
}

class G {
public:
  int epoll_create1(int flags);
  void d() {
    epoll_create1(EPOLL_CLOEXEC);
    TEMP_FAILURE_RETRY(epoll_create1(EPOLL_CLOEXEC));
  }
};
