// RUN: %clang_cc1 -verify -triple x86_64-apple-darwin -emit-llvm -o - %s -std=c++1y | FileCheck %s
// expected-no-diagnostics

struct A {
  constexpr A() : n(1) {}
  ~A();
  int n;
};
struct B : A {
  A a[3];
  constexpr B() {
    ++a[0].n;
    a[1].n += 2;
    a[2].n = n + a[1].n;
  }
};
B b;

// CHECK: @b ={{.*}} global {{.*}} i32 1, {{.*}} { i32 2 }, {{.*}} { i32 3 }, {{.*}} { i32 4 }
// CHECK-NOT: _ZN1BC

namespace ModifyStaticTemporary {
  struct A { int &&temporary; int x; };
  constexpr int f(int &r) { r *= 9; return r - 12; }
  A a = { 6, f(a.temporary) };
  // CHECK: @_ZGRN21ModifyStaticTemporary1aE_ = internal global i32 54
  // CHECK: @_ZN21ModifyStaticTemporary1aE ={{.*}} global {{.*}} ptr @_ZGRN21ModifyStaticTemporary1aE_, i32 42

  A b = { 7, ++b.temporary };
  // CHECK: @_ZGRN21ModifyStaticTemporary1bE_ = internal global i32 8
  // CHECK: @_ZN21ModifyStaticTemporary1bE ={{.*}} global {{.*}} ptr @_ZGRN21ModifyStaticTemporary1bE_, i32 8

  // Can't emit all of 'c' as a constant here, so emit the initial value of
  // 'c.temporary', not the value as modified by the partial evaluation within
  // the initialization of 'c.x'.
  A c = { 10, (++c.temporary, b.x) };
  // CHECK: @_ZN21ModifyStaticTemporary1cE ={{.*}} global {{.*}} zeroinitializer
  // CHECK: @_ZGRN21ModifyStaticTemporary1cE_ = internal global i32 10
}

// CHECK: @_ZGRN28VariableTemplateWithConstRef1iIvEE_ = linkonce_odr constant i32 5, align 4
// CHECK: @_ZN28VariableTemplateWithConstRef3useE ={{.*}} constant ptr @_ZGRN28VariableTemplateWithConstRef1iIvEE_
namespace VariableTemplateWithConstRef {
  template <typename T>
  const int &i = 5;
  const int &use = i<void>;
}

// CHECK: @_ZGRN34HiddenVariableTemplateWithConstRef1iIvEE_ = linkonce_odr hidden constant i32 5, align 4
// CHECK: @_ZN34HiddenVariableTemplateWithConstRef3useE ={{.*}} constant ptr @_ZGRN34HiddenVariableTemplateWithConstRef1iIvEE_
namespace HiddenVariableTemplateWithConstRef {
  template <typename T>
  __attribute__((visibility("hidden"))) const int &i = 5;
  const int &use = i<void>;
}

// CHECK: @_ZGRN24VariableTemplateWithPack1sIJLi1ELi2ELi3ELi4EEEE1_ = linkonce_odr constant i32 1
// CHECK: @_ZGRN24VariableTemplateWithPack1sIJLi1ELi2ELi3ELi4EEEE0_ = linkonce_odr global {{.*}} { ptr @_ZGRN24VariableTemplateWithPack1sIJLi1ELi2ELi3ELi4EEEE1_ }
// CHECK: @_ZGRN24VariableTemplateWithPack1sIJLi1ELi2ELi3ELi4EEEE3_ = linkonce_odr constant i32 2
// CHECK: @_ZGRN24VariableTemplateWithPack1sIJLi1ELi2ELi3ELi4EEEE2_ = linkonce_odr global {{.*}} { ptr @_ZGRN24VariableTemplateWithPack1sIJLi1ELi2ELi3ELi4EEEE3_ }
// CHECK: @_ZGRN24VariableTemplateWithPack1sIJLi1ELi2ELi3ELi4EEEE5_ = linkonce_odr constant i32 3
// CHECK: @_ZGRN24VariableTemplateWithPack1sIJLi1ELi2ELi3ELi4EEEE4_ = linkonce_odr global {{.*}} { ptr @_ZGRN24VariableTemplateWithPack1sIJLi1ELi2ELi3ELi4EEEE5_ }
// CHECK: @_ZGRN24VariableTemplateWithPack1sIJLi1ELi2ELi3ELi4EEEE7_ = linkonce_odr constant i32 4
// CHECK: @_ZGRN24VariableTemplateWithPack1sIJLi1ELi2ELi3ELi4EEEE6_ = linkonce_odr global {{.*}} { ptr @_ZGRN24VariableTemplateWithPack1sIJLi1ELi2ELi3ELi4EEEE7_ }
// CHECK: @_ZGRN24VariableTemplateWithPack1sIJLi1ELi2ELi3ELi4EEEE_ = linkonce_odr global %"struct.VariableTemplateWithPack::S" { ptr @_ZGRN24VariableTemplateWithPack1sIJLi1ELi2ELi3ELi4EEEE0_, ptr @_ZGRN24VariableTemplateWithPack1sIJLi1ELi2ELi3ELi4EEEE2_, ptr @_ZGRN24VariableTemplateWithPack1sIJLi1ELi2ELi3ELi4EEEE4_, ptr @_ZGRN24VariableTemplateWithPack1sIJLi1ELi2ELi3ELi4EEEE6_ }
// CHECK: @_ZN24VariableTemplateWithPack1pE ={{.*}} global {{.*}} @_ZGRN24VariableTemplateWithPack1sIJLi1ELi2ELi3ELi4EEEE_
namespace VariableTemplateWithPack {
  struct A {
    const int &r;
  };
  struct S {
    A &&a, &&b, &&c, &&d;
  };
  template <int... N>
  S &&s = {A{N}...};
  S *p = &s<1, 2, 3, 4>;
}


// CHECK: @_ZGR1z_ ={{.*}} global [2 x i32] [i32 10, i32 2]
// CHECK: @z = global { ptr, i32 } { ptr @_ZGR1z_, i32 10 }
typedef int v[2];
struct Z { int &&x, y; };
Z z = { v{1,2}[0], z.x = 10 };

// CHECK: @_ZGR2z2_ ={{.*}} global %struct.R { i64 10 }
// @z = {{.}} global %struct.Z { ptr @_ZGR1z_, %struct.R { i64 10 } }
struct R { mutable long x; };
struct Z2 { const R &x, y; };
Z2 z2 = { R{1}, z2.x.x = 10 };

// CHECK: __cxa_atexit({{.*}} @_ZN1BD1Ev, {{.*}} @b

// CHECK: define
// CHECK-NOT: @_ZGRN21ModifyStaticTemporary1cE_
// CHECK: store {{.*}} @_ZGRN21ModifyStaticTemporary1cE_, {{.*}} @_ZN21ModifyStaticTemporary1cE
// CHECK: add
// CHECK: store
// CHECK: load {{.*}} @_ZN21ModifyStaticTemporary1bE
// CHECK: store {{.*}} @_ZN21ModifyStaticTemporary1cE
