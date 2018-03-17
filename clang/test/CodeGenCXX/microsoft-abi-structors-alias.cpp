// RUN: %clang_cc1 -emit-llvm %s -o - -triple=i386-pc-win32 -fno-rtti -mconstructor-aliases -O1 -disable-llvm-passes | FileCheck %s

namespace test1 {
template <typename T> class A {
  ~A() {}
};
template class A<char>;
// CHECK-DAG: define weak_odr dso_local x86_thiscallcc void @"??1?$A@D@test1@@AAE@XZ"
}

namespace test2 {
struct A {
  virtual ~A();
};
struct B : A {
  B();
  virtual ~B();
};

A::~A() {}
B::~B() {}
void foo() {
  B b;
}
// CHECK-DAG: @"??1B@test2@@UAE@XZ" = dso_local alias void (%"struct.test2::B"*), bitcast (void (%"struct.test2::A"*)* @"??1A@test2@@UAE@XZ" to void (%"struct.test2::B"*)*)
}

namespace test3 {
struct A { virtual ~A(); };
A::~A() {}
}
// CHECK-DAG: define dso_local x86_thiscallcc void @"??1A@test3@@UAE@XZ"(
namespace test3 {
template <typename T>
struct B : A {
  virtual ~B() { }
};
template struct B<int>;
}
// This has to be weak, and emitting weak aliases is fragile, so we don't do the
// aliasing.
// CHECK-DAG: define weak_odr dso_local x86_thiscallcc void @"??1?$B@H@test3@@UAE@XZ"(
