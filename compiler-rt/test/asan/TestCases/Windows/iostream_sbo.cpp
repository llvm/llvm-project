// UNSUPPORTED: target={{.*-windows-gnu}}

// First, check this works with the default ignorelist:
// RUN: %clang_cl_asan -Od %s -Fe%t
// RUN: echo "42" | %run %t 2>&1 | FileCheck %s
//
// Then, make sure it still works when a user uses their own ignorelist file:
// RUN: %clang_cl_asan -Od %s -fsanitize-ignorelist=%p/../Helpers/initialization-ignorelist.txt -Fe%t2
// RUN: echo "42" | %run %t2 2>&1 | FileCheck %s

#include <iostream>

int main() {
  int i;
  std::cout << "Type i: ";
  std::cin >> i;
  return 0;
// CHECK: Type i:
// CHECK-NOT: AddressSanitizer: stack-buffer-overflow on address [[ADDR:0x[0-9a-f]+]]
}
