// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++23 %s -emit-llvm -o - | FileCheck %s

consteval const int* a() {
    static constexpr int b = 3;
    return &b;
}

consteval const int returns42() {
    static constexpr long c = 42;
    return c;
}

struct S {
  int a;
  constexpr S(int _a) : a(_a){};
};

consteval auto returns48viastruct() {
    static constexpr S s{ 48 };
    return s;
}

consteval int why() {
  static constexpr int a = 10;
  {
  static constexpr int a = 20;
  return a;
  }

}

consteval const int* returnsarray() {
  static constexpr int a[] = {10, 20, 30};
  return a;
}

// CHECK: @_ZZ1avE1b = linkonce_odr constant i32 3, comdat, align 4
// CHECK: @_ZZ12returnsarrayvE1a = linkonce_odr constant [3 x i32] [i32 10, i32 20, i32 30], comdat, align 4

// CHECK: define dso_local noundef i32 @_Z1cv()
// CHECK-NEXT: entry:
// CHECK-NEXT:   %0 = load i32, ptr @_ZZ1avE1b, align 4
// CHECK-NEXT:   ret i32 %0
// CHECK-NEXT: }
int c() { return *a(); }

// CHECK: define dso_local noundef i32 @_Z1dv()
// CHECK-NEXT: entry:
// CHECK-NEXT:   ret i32 42
// CHECK-NEXT: }
int d() { return returns42(); }

int e() { return returns48viastruct().a; }
// CHECK: define dso_local noundef i32 @_Z1ev()
// CHECK-NEXT: entry:
// CHECK-NEXT:   %ref.tmp = alloca %struct.S, align 4
// CHECK-NEXT:   %0 = getelementptr inbounds nuw %struct.S, ptr %ref.tmp, i32 0, i32 0
// CHECK-NEXT:   store i32 48, ptr %0, align 4
// CHECK-NEXT:   ret i32 48
// CHECK-NEXT: }

int f() { return why(); }
// CHECK: define dso_local noundef i32 @_Z1fv()
// CHECK-NEXT: entry:
// CHECK-NEXT:   ret i32 20
// CHECK-NEXT: }

int g() { return returnsarray()[2]; }
// CHECK: define dso_local noundef i32 @_Z1gv()
// CHECK-NEXT: entry:
// CHECK-NEXT:   %0 = load i32, ptr getelementptr inbounds (i32, ptr @_ZZ12returnsarrayvE1a, i64 2), align 4
// CHECK-NEXT:   ret i32 %0
// CHECK-NEXT: }
