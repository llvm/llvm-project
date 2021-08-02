// RUN: %clangxx_asan %stdcxx11 -O1 -fsanitize-address-use-after-scope %s -o %t && \
// RUN:     not %run %t 2>&1 | FileCheck %s
//
// Not expected to work yet with HWAsan.
// XFAIL: *

struct IntHolder {
  __attribute__((noinline)) const IntHolder &Self() const {
    return *this;
  }
  int val = 3;
};

const IntHolder *saved;

int main(int argc, char *argv[]) {
  saved = &IntHolder().Self();
  int x = saved->val; // BOOM
  // CHECK: ERROR: AddressSanitizer: stack-use-after-scope
  // CHECK:  #0 0x{{.*}} in main {{.*}}use-after-scope-temp2.cpp:[[@LINE-2]]
  return x;
}
