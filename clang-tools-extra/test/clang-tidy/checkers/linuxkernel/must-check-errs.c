// RUN: %check_clang_tidy %s linuxkernel-must-check-errs %t

#define __must_check __attribute__((warn_unused_result))

// Prototypes of the error functions.
void * __must_check ERR_PTR(long error);
long  __must_check PTR_ERR(const void *ptr);
int  __must_check IS_ERR(const void *ptr);
int  __must_check IS_ERR_OR_NULL(const void *ptr);
void * __must_check ERR_CAST(const void *ptr);
int  __must_check PTR_ERR_OR_ZERO(const void *ptr);

void f(void) {
  ERR_PTR(0);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: result from function 'ERR_PTR' is unused
  PTR_ERR((void *)0);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: result from function 'PTR_ERR' is unused
  IS_ERR((void *)0);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: result from function 'IS_ERR' is unused
  ERR_CAST((void *)0);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: result from function 'ERR_CAST' is unused
out:
  PTR_ERR_OR_ZERO((void *)0);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: result from function 'PTR_ERR_OR_ZERO' is unused
}

void *f1(void) {
  return ERR_PTR(0);
}

long f2(void) {
  if (IS_ERR((void *)0)) {
    return PTR_ERR((void *)0);
  }
  return -1;
}

void f3(void) {
  f1();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: result from function 'f1' is unused but represents an error value
  f2();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: result from function 'f2' is unused but represents an error value
}
