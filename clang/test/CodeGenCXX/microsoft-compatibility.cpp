// RUN: %clang_cc1 -no-opaque-pointers %s -triple i686-pc-win32 -std=c++11 -fms-compatibility -emit-llvm -o - | FileCheck %s

template <typename>
struct S {
  static const int x[];
};

template <>
const int S<char>::x[] = {1};

// CHECK-LABEL: @"?x@?$S@D@@2QBHB" = weak_odr dso_local constant [1 x i32] [i32 1], comdat

template<class T>
void destroy(T *p) {
  p->~T();
}

extern "C" void f() {
  int a;
  destroy((void*)&a);
}

// CHECK-LABEL: define dso_local void @f()
// CHECK: call void @"??$destroy@X@@YAXPAX@Z"
// CHECK: ret void

// CHECK-LABEL: define linkonce_odr dso_local void @"??$destroy@X@@YAXPAX@Z"(i8* noundef %p)
//    The pseudo-dtor expr should not generate calls to anything.
// CHECK-NOT: call
// CHECK-NOT: invoke
// CHECK: ret void
