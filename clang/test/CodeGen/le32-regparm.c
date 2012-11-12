// RUN: %clang_cc1 -triple le32-unknown-nacl %s -emit-llvm -o - | FileCheck %s

#define FASTCALL __attribute__((regparm(2)))

typedef struct {
  int aaa;
  double bbbb;
  int ccc[200];
} foo;

// 2 inreg arguments are supported.
void FASTCALL f1(int i, int j, int k);
// CHECK: define void @f1(i32 inreg %i, i32 inreg %j, i32 %k)
void f1(int i, int j, int k) { }

// inreg structs are not supported.
// CHECK: define void @f2(%struct.foo* inreg %a)
void __attribute__((regparm(1))) f2(foo* a) {}

// Only the first 2 arguments can be passed inreg, and the first
// non-integral type consumes remaining available registers.
// CHECK: define void @f3(%struct.foo* byval %a, i32 %b)
void __attribute__((regparm(2))) f3(foo a, int b) {}

// Only 64 total bits are supported
// CHECK: define void @f4(i64 inreg %g, i32 %h)
void __attribute__((regparm(2))) f4(long long g, int h) {}

typedef void (*FType)(int, int) __attribute ((regparm (2)));
FType bar;
extern void FASTCALL reduced(char b, double c, foo* d, double e, int f);

int
main(void) {
  // The presence of double c means that foo* d is not passed inreg. This
  // behavior is different from current x86-32 behavior
  // CHECK: call void @reduced(i8 signext inreg 0, {{.*}} %struct.foo* null
  reduced(0, 0.0, 0, 0.0, 0);
  // CHECK: call void {{.*}}(i32 inreg 1, i32 inreg 2)
  bar(1,2);
}
