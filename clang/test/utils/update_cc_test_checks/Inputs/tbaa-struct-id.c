// RUN: %clang_cc1 -flegacy-pass-manager -triple arm64-apple-iphoneos -Oz -mllvm -enable-constraint-elimination=true -fpass-by-value-is-noalias -emit-llvm -o - %s | FileCheck %s

typedef struct {
  void *a;
  void *b;
  void *c;
  void *d;
  void *e;
} Foo;

static void bar(Foo f) {
  if (f.b)
    __builtin_trap();
}

static int baz(Foo f) {
  bar(f);
  return *(int *)f.a;
}

int barbar(Foo arg) {
  int a, b;
  a = baz(arg);
  return a - b;
}
