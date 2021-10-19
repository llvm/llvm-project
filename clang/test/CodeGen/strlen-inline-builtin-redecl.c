// RUN: %clang_cc1 -triple x86_64 -S -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s
//
// Verifies that clang-generated *.inline are removed when shadowed by an external definition

// CHECK-NOT: strlen.inline

unsigned long strnlen(const char *, unsigned long);
void fortify_panic(const char *);

extern inline __attribute__((always_inline)) __attribute__((gnu_inline)) unsigned long strlen(const char *p) {
  return 1;
}
unsigned long mystrlen(char const *s) {
  return strlen(s);
}
unsigned long strlen(const char *s) {
  return 2;
}
unsigned long yourstrlen(char const *s) {
  return strlen(s);
}
