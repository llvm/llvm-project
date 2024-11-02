// RUN: %check_clang_tidy %s android-cloexec-memfd-create %t

#define MFD_ALLOW_SEALING 1
#define __O_CLOEXEC 3
#define MFD_CLOEXEC __O_CLOEXEC
#define TEMP_FAILURE_RETRY(exp) \
  ({                            \
    int _rc;                    \
    do {                        \
      _rc = (exp);              \
    } while (_rc == -1);        \
  })
#define NULL 0

extern "C" int memfd_create(const char *name, unsigned int flags);

void a() {
  memfd_create(NULL, MFD_ALLOW_SEALING);
  // CHECK-MESSAGES: :[[@LINE-1]]:39: warning: 'memfd_create' should use MFD_CLOEXEC where possible [android-cloexec-memfd-create]
  // CHECK-FIXES: memfd_create(NULL, MFD_ALLOW_SEALING | MFD_CLOEXEC)
  TEMP_FAILURE_RETRY(memfd_create(NULL, MFD_ALLOW_SEALING));
  // CHECK-MESSAGES: :[[@LINE-1]]:58: warning: 'memfd_create'
  // CHECK-FIXES: TEMP_FAILURE_RETRY(memfd_create(NULL, MFD_ALLOW_SEALING | MFD_CLOEXEC))
}

void f() {
  memfd_create(NULL, 3);
  // CHECK-MESSAGES: :[[@LINE-1]]:23: warning: 'memfd_create'
  // CHECK-FIXES: memfd_create(NULL, 3 | MFD_CLOEXEC)
  TEMP_FAILURE_RETRY(memfd_create(NULL, 3));
  // CHECK-MESSAGES: :[[@LINE-1]]:42: warning: 'memfd_create'
  // CHECK-FIXES: TEMP_FAILURE_RETRY(memfd_create(NULL, 3 | MFD_CLOEXEC))

  int flag = 3;
  memfd_create(NULL, flag);
  TEMP_FAILURE_RETRY(memfd_create(NULL, flag));
}

namespace i {
int memfd_create(const char *name, unsigned int flags);

void d() {
  memfd_create(NULL, MFD_ALLOW_SEALING);
  TEMP_FAILURE_RETRY(memfd_create(NULL, MFD_ALLOW_SEALING));
}

} // namespace i

void e() {
  memfd_create(NULL, MFD_CLOEXEC);
  TEMP_FAILURE_RETRY(memfd_create(NULL, MFD_CLOEXEC));
  memfd_create(NULL, MFD_ALLOW_SEALING | MFD_CLOEXEC);
  TEMP_FAILURE_RETRY(memfd_create(NULL, MFD_ALLOW_SEALING | MFD_CLOEXEC));
}

class G {
public:
  int memfd_create(const char *name, unsigned int flags);
  void d() {
    memfd_create(NULL, MFD_ALLOW_SEALING);
    TEMP_FAILURE_RETRY(memfd_create(NULL, MFD_ALLOW_SEALING));
  }
};
