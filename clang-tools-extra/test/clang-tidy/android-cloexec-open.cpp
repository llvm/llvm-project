// RUN: %check_clang_tidy %s android-cloexec-open %t

#define O_RDWR 1
#define O_EXCL 2
#define __O_CLOEXEC 3
#define O_CLOEXEC __O_CLOEXEC
#define TEMP_FAILURE_RETRY(exp) \
  ({                            \
    int _rc;                    \
    do {                        \
      _rc = (exp);              \
    } while (_rc == -1);        \
  })

extern "C" int open(const char *fn, int flags, ...);
extern "C" int open64(const char *fn, int flags, ...);
extern "C" int openat(int dirfd, const char *pathname, int flags, ...);

void a() {
  open("filename", O_RDWR);
  // CHECK-MESSAGES: :[[@LINE-1]]:26: warning: 'open' should use O_CLOEXEC where possible [android-cloexec-open]
  // CHECK-FIXES: O_RDWR | O_CLOEXEC
  TEMP_FAILURE_RETRY(open("filename", O_RDWR));
  // CHECK-MESSAGES: :[[@LINE-1]]:45: warning: 'open' should use O_CLOEXEC where
  // CHECK-FIXES: O_RDWR | O_CLOEXEC
  open("filename", O_RDWR | O_EXCL);
  // CHECK-MESSAGES: :[[@LINE-1]]:35: warning: 'open' should use O_CLOEXEC where
  // CHECK-FIXES: O_RDWR | O_EXCL | O_CLOEXEC
  TEMP_FAILURE_RETRY(open("filename", O_RDWR | O_EXCL));
  // CHECK-MESSAGES: :[[@LINE-1]]:54: warning: 'open' should use O_CLOEXEC where
  // CHECK-FIXES: O_RDWR | O_EXCL | O_CLOEXEC
}

void b() {
  open64("filename", O_RDWR);
  // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: 'open64' should use O_CLOEXEC where possible [android-cloexec-open]
  // CHECK-FIXES: O_RDWR | O_CLOEXEC
  TEMP_FAILURE_RETRY(open64("filename", O_RDWR));
  // CHECK-MESSAGES: :[[@LINE-1]]:47: warning: 'open64' should use O_CLOEXEC where
  // CHECK-FIXES: O_RDWR | O_CLOEXEC
  open64("filename", O_RDWR | O_EXCL);
  // CHECK-MESSAGES: :[[@LINE-1]]:37: warning: 'open64' should use O_CLOEXEC where
  // CHECK-FIXES: O_RDWR | O_EXCL | O_CLOEXEC
  TEMP_FAILURE_RETRY(open64("filename", O_RDWR | O_EXCL));
  // CHECK-MESSAGES: :[[@LINE-1]]:56: warning: 'open64' should use O_CLOEXEC where
  // CHECK-FIXES: O_RDWR | O_EXCL | O_CLOEXEC
}

void c() {
  openat(0, "filename", O_RDWR);
  // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: 'openat' should use O_CLOEXEC where possible [android-cloexec-open]
  // CHECK-FIXES: O_RDWR | O_CLOEXEC
  TEMP_FAILURE_RETRY(openat(0, "filename", O_RDWR));
  // CHECK-MESSAGES: :[[@LINE-1]]:50: warning: 'openat' should use O_CLOEXEC where
  // CHECK-FIXES: O_RDWR | O_CLOEXEC
  openat(0, "filename", O_RDWR | O_EXCL);
  // CHECK-MESSAGES: :[[@LINE-1]]:40: warning: 'openat' should use O_CLOEXEC where
  // CHECK-FIXES: O_RDWR | O_EXCL | O_CLOEXEC
  TEMP_FAILURE_RETRY(openat(0, "filename", O_RDWR | O_EXCL));
  // CHECK-MESSAGES: :[[@LINE-1]]:59: warning: 'openat' should use O_CLOEXEC where
  // CHECK-FIXES: O_RDWR | O_EXCL | O_CLOEXEC
}

void f() {
  open("filename", 3);
  // CHECK-MESSAGES: :[[@LINE-1]]:21: warning: 'open' should use O_CLOEXEC where possible [android-cloexec-open]
  // CHECK-FIXES: 3 | O_CLOEXEC
  TEMP_FAILURE_RETRY(open("filename", 3));
  // CHECK-MESSAGES: :[[@LINE-1]]:40: warning: 'open' should use O_CLOEXEC where
  // CHECK-FIXES: 3 | O_CLOEXEC
  open64("filename", 3);
  // CHECK-MESSAGES: :[[@LINE-1]]:23: warning: 'open64' should use O_CLOEXEC where possible [android-cloexec-open]
  // CHECK-FIXES: 3 | O_CLOEXEC
  TEMP_FAILURE_RETRY(open64("filename", 3));
  // CHECK-MESSAGES: :[[@LINE-1]]:42: warning: 'open64' should use O_CLOEXEC where
  // CHECK-FIXES: 3 | O_CLOEXEC
  openat(0, "filename", 3);
  // CHECK-MESSAGES: :[[@LINE-1]]:26: warning: 'openat' should use O_CLOEXEC where possible [android-cloexec-open]
  // CHECK-FIXES: 3 | O_CLOEXEC
  TEMP_FAILURE_RETRY(openat(0, "filename", 3));
  // CHECK-MESSAGES: :[[@LINE-1]]:45: warning: 'openat' should use O_CLOEXEC where
  // CHECK-FIXES: 3 | O_CLOEXEC

  int flag = 3;
  open("filename", flag);
  // CHECK-MESSAGES-NOT: warning:
  TEMP_FAILURE_RETRY(open("filename", flag));
  // CHECK-MESSAGES-NOT: warning:
  open64("filename", flag);
  // CHECK-MESSAGES-NOT: warning:
  TEMP_FAILURE_RETRY(open64("filename", flag));
  // CHECK-MESSAGES-NOT: warning:
  openat(0, "filename", flag);
  // CHECK-MESSAGES-NOT: warning:
  TEMP_FAILURE_RETRY(openat(0, "filename", flag));
  // CHECK-MESSAGES-NOT: warning:
}

namespace i {
int open(const char *pathname, int flags, ...);
int open64(const char *pathname, int flags, ...);
int openat(int dirfd, const char *pathname, int flags, ...);

void d() {
  open("filename", O_RDWR);
  // CHECK-MESSAGES-NOT: warning:
  TEMP_FAILURE_RETRY(open("filename", O_RDWR));
  // CHECK-MESSAGES-NOT: warning:
  open64("filename", O_RDWR);
  // CHECK-MESSAGES-NOT: warning:
  TEMP_FAILURE_RETRY(open64("filename", O_RDWR));
  // CHECK-MESSAGES-NOT: warning:
  openat(0, "filename", O_RDWR);
  // CHECK-MESSAGES-NOT: warning:
  TEMP_FAILURE_RETRY(openat(0, "filename", O_RDWR));
  // CHECK-MESSAGES-NOT: warning:
}

} // namespace i

void e() {
  open("filename", O_CLOEXEC);
  // CHECK-MESSAGES-NOT: warning:
  TEMP_FAILURE_RETRY(open("filename", O_CLOEXEC));
  // CHECK-MESSAGES-NOT: warning:
  open("filename", O_RDWR | O_CLOEXEC);
  // CHECK-MESSAGES-NOT: warning:
  TEMP_FAILURE_RETRY(open("filename", O_RDWR | O_CLOEXEC));
  // CHECK-MESSAGES-NOT: warning:
  open("filename", O_RDWR | O_CLOEXEC | O_EXCL);
  // CHECK-MESSAGES-NOT: warning:
  TEMP_FAILURE_RETRY(open("filename", O_RDWR | O_CLOEXEC | O_EXCL));
  // CHECK-MESSAGES-NOT: warning:
  open64("filename", O_CLOEXEC);
  // CHECK-MESSAGES-NOT: warning:
  TEMP_FAILURE_RETRY(open64("filename", O_CLOEXEC));
  // CHECK-MESSAGES-NOT: warning:
  open64("filename", O_RDWR | O_CLOEXEC);
  // CHECK-MESSAGES-NOT: warning:
  TEMP_FAILURE_RETRY(open64("filename", O_RDWR | O_CLOEXEC));
  // CHECK-MESSAGES-NOT: warning:
  open64("filename", O_RDWR | O_CLOEXEC | O_EXCL);
  // CHECK-MESSAGES-NOT: warning:
  TEMP_FAILURE_RETRY(open64("filename", O_RDWR | O_CLOEXEC | O_EXCL));
  // CHECK-MESSAGES-NOT: warning:
  openat(0, "filename", O_CLOEXEC);
  // CHECK-MESSAGES-NOT: warning:
  TEMP_FAILURE_RETRY(openat(0, "filename", O_CLOEXEC));
  // CHECK-MESSAGES-NOT: warning:
  openat(0, "filename", O_RDWR | O_CLOEXEC);
  // CHECK-MESSAGES-NOT: warning:
  TEMP_FAILURE_RETRY(openat(0, "filename", O_RDWR | O_CLOEXEC));
  // CHECK-MESSAGES-NOT: warning:
  openat(0, "filename", O_RDWR | O_CLOEXEC | O_EXCL);
  // CHECK-MESSAGES-NOT: warning:
  TEMP_FAILURE_RETRY(openat(0, "filename", O_RDWR | O_CLOEXEC | O_EXCL));
  // CHECK-MESSAGES-NOT: warning:
}

class G {
public:
  int open(const char *pathname, int flags, ...);
  int open64(const char *pathname, int flags, ...);
  int openat(int dirfd, const char *pathname, int flags, ...);

  void h() {
    open("filename", O_RDWR);
    // CHECK-MESSAGES-NOT: warning:
    TEMP_FAILURE_RETRY(open("filename", O_RDWR));
    // CHECK-MESSAGES-NOT: warning:
    open64("filename", O_RDWR);
    // CHECK-MESSAGES-NOT: warning:
    TEMP_FAILURE_RETRY(open64("filename", O_RDWR));
    // CHECK-MESSAGES-NOT: warning:
    openat(0, "filename", O_RDWR);
    // CHECK-MESSAGES-NOT: warning:
    TEMP_FAILURE_RETRY(openat(0, "filename", O_RDWR));
    // CHECK-MESSAGES-NOT: warning:
  }
};
