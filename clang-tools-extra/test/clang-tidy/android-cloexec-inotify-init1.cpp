// RUN: %check_clang_tidy %s android-cloexec-inotify-init1 %t

#define IN_NONBLOCK 1
#define __O_CLOEXEC 3
#define IN_CLOEXEC __O_CLOEXEC
#define TEMP_FAILURE_RETRY(exp) \
  ({                            \
    int _rc;                    \
    do {                        \
      _rc = (exp);              \
    } while (_rc == -1);        \
  })

extern "C" int inotify_init1(int flags);

void a() {
  inotify_init1(IN_NONBLOCK);
  // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: 'inotify_init1' should use IN_CLOEXEC where possible [android-cloexec-inotify-init1]
  // CHECK-FIXES: inotify_init1(IN_NONBLOCK | IN_CLOEXEC);
  TEMP_FAILURE_RETRY(inotify_init1(IN_NONBLOCK));
  // CHECK-MESSAGES: :[[@LINE-1]]:47: warning: 'inotify_init1'
  // CHECK-FIXES: TEMP_FAILURE_RETRY(inotify_init1(IN_NONBLOCK | IN_CLOEXEC));
}

void f() {
  inotify_init1(0);
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: 'inotify_init1'
  // CHECK-FIXES: inotify_init1(IN_CLOEXEC);
  TEMP_FAILURE_RETRY(inotify_init1(0));
  // CHECK-MESSAGES: :[[@LINE-1]]:37: warning: 'inotify_init1'
  // CHECK-FIXES: TEMP_FAILURE_RETRY(inotify_init1(IN_CLOEXEC));

  int flag = 1;
  inotify_init1(flag);
  TEMP_FAILURE_RETRY(inotify_init1(flag));
}

namespace i {
int inotify_init1(int flags);

void d() {
  inotify_init1(IN_NONBLOCK);
  TEMP_FAILURE_RETRY(inotify_init1(IN_NONBLOCK));
}

} // namespace i

void e() {
  inotify_init1(IN_CLOEXEC);
  TEMP_FAILURE_RETRY(inotify_init1(IN_CLOEXEC));
  inotify_init1(IN_NONBLOCK | IN_CLOEXEC);
  TEMP_FAILURE_RETRY(inotify_init1(IN_NONBLOCK | IN_CLOEXEC));
}

class G {
public:
  int inotify_init1(int flags);
  void d() {
    inotify_init1(IN_CLOEXEC);
    TEMP_FAILURE_RETRY(inotify_init1(IN_CLOEXEC));
    inotify_init1(IN_NONBLOCK | IN_CLOEXEC);
    TEMP_FAILURE_RETRY(inotify_init1(IN_NONBLOCK | IN_CLOEXEC));
  }
};
