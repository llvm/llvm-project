// RUN: %clang_cc1 -triple=x86_64-unknown-linux-gnu -emit-llvm -o - %s | FileCheck -check-prefixes=CHECK %s
// RUN: %clang_cc1 -triple=x86_64-unknown-linux-gnu -emit-llvm -o - %s -DFOO | FileCheck -check-prefixes=CHECK,FOO %s

#ifdef FOO
int foo() {
  return 1;
}
#endif

int bar() {
  return 2;
}
