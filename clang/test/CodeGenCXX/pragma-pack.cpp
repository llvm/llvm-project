// RUN: %clang_cc1 %s -triple=i686-apple-darwin10 -emit-llvm -o - | FileCheck %s

struct Base {
  virtual ~Base();
  int x;
};

#pragma pack(1)
struct Sub : virtual Base {
  char c;
};

// CHECK: %struct.Sub = type <{ ptr, i8, %struct.Base }>
Sub f(Sub x) {
  return x;
}

static int i[sizeof(Sub) == 13 ? 1 : -1];
