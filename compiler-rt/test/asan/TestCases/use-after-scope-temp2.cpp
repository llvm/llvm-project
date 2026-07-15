// RUN: %clangxx_asan %if MSVC %{ /Od %} %else %{ -O1 %} \
// RUN:     %s -o %t && not %run %t 2>&1 | FileCheck %s

#include "defines.h"

struct IntHolder {
  ATTRIBUTE_NOINLINE const IntHolder &Self() const { return *this; }
  int val = 3;
};

const IntHolder *saved;

int main(int argc, char *argv[]) {
  saved = &IntHolder().Self();
  int x = saved->val;  // BOOM
  // CHECK: ERROR: AddressSanitizer: stack-use-after-scope
  // CHECK:  #0 0x{{.*}} in main {{.*}}use-after-scope-temp2.cpp:[[@LINE-2]]
  return x;
}
