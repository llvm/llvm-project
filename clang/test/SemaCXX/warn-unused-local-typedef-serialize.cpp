// RUN: %clang -x c++-header -c -Wunused-local-typedef %s -o %t.pch -Werror
// RUN: %clang -DBE_THE_SOURCE -c -Wunused-local-typedef -include %t %s -o /dev/null 2>&1 | FileCheck %s
// RUN: %clang -DBE_THE_SOURCE -c -Wunused-local-typedef -include %t %s -o /dev/null 2>&1 | FileCheck %s

#ifndef BE_THE_SOURCE
inline void myfun() {
// The warning should fire every time the pch file is used, not when it's built.
// CHECK: warning: unused typedef
  typedef int a;
}
#endif
