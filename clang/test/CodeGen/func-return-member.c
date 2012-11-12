// RUN: %clang_cc1 -triple x86_64-apple-darwin -emit-llvm -o - %s | FileCheck %s

struct frk { float _Complex c; int x; };
struct faz { struct frk f; };
struct fuz { struct faz f; };

extern struct fuz foo(void);

int X;
struct frk F;
float _Complex C;

// CHECK: define void @bar
void bar(void) {
  X = foo().f.f.x;
}

// CHECK: define void @bun
void bun(void) {
  F = foo().f.f;
}

// CHECK: define void @ban
void ban(void) {
  C = foo().f.f.c;
}
