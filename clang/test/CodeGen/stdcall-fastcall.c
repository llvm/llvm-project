// RUN: %clang_cc1 -no-opaque-pointers -triple i386-unknown-unknown -Wno-strict-prototypes -emit-llvm < %s | FileCheck %s

void __attribute__((fastcall)) f1(void);
void __attribute__((stdcall)) f2(void);
void __attribute__((thiscall)) f3(void);
void __attribute__((fastcall)) f4(void) {
// CHECK-LABEL: define{{.*}} x86_fastcallcc void @f4()
  f1();
// CHECK: call x86_fastcallcc void @f1()
}
void __attribute__((stdcall)) f5(void) {
// CHECK-LABEL: define{{.*}} x86_stdcallcc void @f5()
  f2();
// CHECK: call x86_stdcallcc void @f2()
}
void __attribute__((thiscall)) f6(void) {
// CHECK-LABEL: define{{.*}} x86_thiscallcc void @f6()
  f3();
// CHECK: call x86_thiscallcc void @f3()
}

// PR5280
void (__attribute__((fastcall)) *pf1)(void) = f1;
void (__attribute__((stdcall)) *pf2)(void) = f2;
void (__attribute__((thiscall)) *pf3)(void) = f3;
void (__attribute__((fastcall)) *pf4)(void) = f4;
void (__attribute__((stdcall)) *pf5)(void) = f5;
void (__attribute__((thiscall)) *pf6)(void) = f6;

int main(void) {
    f4(); f5(); f6();
    // CHECK: call x86_fastcallcc void @f4()
    // CHECK: call x86_stdcallcc void @f5()
    // CHECK: call x86_thiscallcc void @f6()
    pf1(); pf2(); pf3(); pf4(); pf5(); pf6();
    // CHECK: call x86_fastcallcc void %{{.*}}()
    // CHECK: call x86_stdcallcc void %{{.*}}()
    // CHECK: call x86_thiscallcc void %{{.*}}()
    // CHECK: call x86_fastcallcc void %{{.*}}()
    // CHECK: call x86_stdcallcc void %{{.*}}()
    // CHECK: call x86_thiscallcc void %{{.*}}()
    return 0;
}

// PR7117
void __attribute((stdcall)) f7(foo) int foo; {}
void f8(void) {
  f7(0);
  // CHECK: call x86_stdcallcc void @f7(i32 noundef 0)
}

void __attribute__((fastcall)) foo1(int y);
void bar1(int y) {
  // CHECK-LABEL: define{{.*}} void @bar1
  // CHECK: call x86_fastcallcc void @foo1(i32 inreg noundef %
  foo1(y);
}

struct S1 {
  int x;
};
void __attribute__((fastcall)) foo2(struct S1 y);
void bar2(struct S1 y) {
  // CHECK-LABEL: define{{.*}} void @bar2
  // CHECK: call x86_fastcallcc void @foo2(i32 inreg undef, i32 %
  foo2(y);
}

void __attribute__((fastcall)) foo3(int *y);
void bar3(int *y) {
  // CHECK-LABEL: define{{.*}} void @bar3
  // CHECK: call x86_fastcallcc void @foo3(i32* inreg noundef %
  foo3(y);
}

enum Enum {Eval};
void __attribute__((fastcall)) foo4(enum Enum y);
void bar4(enum Enum y) {
  // CHECK-LABEL: define{{.*}} void @bar4
  // CHECK: call x86_fastcallcc void @foo4(i32 inreg noundef %
  foo4(y);
}

struct S2 {
  int x1;
  double x2;
  double x3;
};
void __attribute__((fastcall)) foo5(struct S2 y);
void bar5(struct S2 y) {
  // CHECK-LABEL: define{{.*}} void @bar5
  // CHECK: call x86_fastcallcc void @foo5(%struct.S2* noundef byval(%struct.S2) align 4 %
  foo5(y);
}

void __attribute__((fastcall)) foo6(long long y);
void bar6(long long y) {
  // CHECK-LABEL: define{{.*}} void @bar6
  // CHECK: call x86_fastcallcc void @foo6(i64 noundef %
  foo6(y);
}

void __attribute__((fastcall)) foo7(int a, struct S1 b, int c);
void bar7(int a, struct S1 b, int c) {
  // CHECK-LABEL: define{{.*}} void @bar7
  // CHECK: call x86_fastcallcc void @foo7(i32 inreg noundef %{{.*}}, i32 %{{.*}}, i32 noundef %{{.*}}
  foo7(a, b, c);
}

void __attribute__((fastcall)) foo8(struct S1 a, int b);
void bar8(struct S1 a, int b) {
  // CHECK-LABEL: define{{.*}} void @bar8
  // CHECK: call x86_fastcallcc void @foo8(i32 inreg undef, i32 %{{.*}}, i32 inreg noundef %
  foo8(a, b);
}

void __attribute__((fastcall)) foo9(struct S2 a, int b);
void bar9(struct S2 a, int b) {
  // CHECK-LABEL: define{{.*}} void @bar9
  // CHECK: call x86_fastcallcc void @foo9(%struct.S2* noundef byval(%struct.S2) align 4 %{{.*}}, i32 noundef %
  foo9(a, b);
}

void __attribute__((fastcall)) foo10(float y, int x);
void bar10(float y, int x) {
  // CHECK-LABEL: define{{.*}} void @bar10
  // CHECK: call x86_fastcallcc void @foo10(float noundef %{{.*}}, i32 inreg noundef %
  foo10(y, x);
}

void __attribute__((fastcall)) foo11(double y, int x);
void bar11(double y, int x) {
  // CHECK-LABEL: define{{.*}} void @bar11
  // CHECK: call x86_fastcallcc void @foo11(double noundef %{{.*}}, i32 inreg noundef %
  foo11(y, x);
}

struct S3 {
  float x;
};
void __attribute__((fastcall)) foo12(struct S3 y, int x);
void bar12(struct S3 y, int x) {
  // CHECK-LABEL: define{{.*}} void @bar12
  // CHECK: call x86_fastcallcc void @foo12(float %{{.*}}, i32 inreg noundef %
  foo12(y, x);
}

void __attribute__((fastcall)) foo13(long long a, int b, int c);
void bar13(long long a, int b, int c) {
  // CHECK-LABEL: define{{.*}} void @bar13
  // CHECK: call x86_fastcallcc void @foo13(i64 noundef %{{.*}}, i32 inreg noundef %{{.*}}, i32 inreg noundef %
  foo13(a, b, c);
}
