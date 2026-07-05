// RUN: %clang_cc1 -triple x86_64 -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s
//
// Verifies that clang-generated *.inline are removed when shadowed by an
// external definition, even when that definition appears at the end of the
// file.

// CHECK-NOT: strlen.inline

extern unsigned long strlen(char const *s);

extern __inline __attribute__((__always_inline__)) __attribute__((__gnu_inline__)) unsigned long strlen(char const *s) {
  return 1;
}

static unsigned long chesterfield(char const *s) {
  return strlen(s);
}
static unsigned long (*_strlen)(char const *ptr);

unsigned long blutch(char const *s) {
  return chesterfield(s);
}

unsigned long strlen(char const *s) {
  return _strlen(s);
}
