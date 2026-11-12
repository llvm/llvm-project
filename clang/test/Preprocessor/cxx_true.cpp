/* RUN: %clang_cc1 -E %s -x c++ | FileCheck -check-prefix CPP %s
   RUN: %clang_cc1 -E %s -x c | FileCheck -check-prefix C %s
   RUN: %clang_cc1 -E %s -x c++ -verify -Wundef
*/
// expected-no-diagnostics

#if true
// CPP: test block_1
// C-NOT: test block_1
test block_1
#endif

#if false
// CPP-NOT: test block_2
// C-NOT: test block_2
test block_2
#endif

