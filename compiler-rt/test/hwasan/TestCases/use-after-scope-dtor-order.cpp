// This is the ASAN test of the same name ported to HWAsan.

// RUN: %clangxx_hwasan -mllvm -hwasan-use-after-scope -O1 %s -o %t && \
// RUN:     not %run %t 2>&1 | FileCheck %s

// REQUIRES: aarch64-target-arch

#include <stdio.h>

struct IntHolder {
  explicit IntHolder(int *val = 0) : val_(val) {}
  __attribute__((noinline)) ~IntHolder() {
    printf("Value: %d\n", *val_); // BOOM
    // CHECK: ERROR: HWAddressSanitizer: tag-mismatch
    // CHECK:  #0 0x{{.*}} in IntHolder::~IntHolder{{.*}}.cpp:[[@LINE-2]]
  }
  void set(int *val) { val_ = val; }
  int *get() { return val_; }

  int *val_;
};

int main(int argc, char *argv[]) {
  // It is incorrect to use "x" int IntHolder destructor, because "x" is
  // "destroyed" earlier as it's declared later.
  IntHolder holder;
  int x = argc;
  holder.set(&x);
  return 0;
}
