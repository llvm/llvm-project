// RUN: %clang_cc1 -emit-llvm -triple %itanium_abi_triple -Werror %s -o - | FileCheck %s

// CHECK:      @llvm.used = appending global [7 x i8*]
// CHECK-SAME:   @_ZN2X0C2Ev
// CHECK-SAME:   @_ZN2X0C1Ev
// CHECK-SAME:   @_ZN2X0D2Ev
// CHECK-SAME:   @_ZN2X0D1Ev
// CHECK-SAME:   @_ZN2X16Nested2f1Ev
// CHECK-SAME:   @_ZN10merge_declL4funcEv
// CHECK-SAME:   @_ZN18instantiate_member1SIiE1fEv

struct X0 {
  // CHECK: define linkonce_odr{{.*}} @_ZN2X0C1Ev({{.*}}
  __attribute__((used, retain)) X0() {}
  // CHECK: define linkonce_odr{{.*}} @_ZN2X0D1Ev({{.*}}
  __attribute__((used, retain)) ~X0() {}
};

struct X1 {
  struct Nested {
    // CHECK-NOT: 2f0Ev
    // CHECK: define linkonce_odr{{.*}} @_ZN2X16Nested2f1Ev({{.*}}
    void __attribute__((retain)) f0() {}
    void __attribute__((used, retain)) f1() {}
  };
};

// CHECK: define internal void @_ZN10merge_declL4funcEv(){{.*}}
namespace merge_decl {
static void func();
void bar() { void func() __attribute__((used, retain)); }
static void func() {}
} // namespace merge_decl

namespace instantiate_member {
template <typename T>
struct S {
  void __attribute__((used, retain)) f() {}
};

void test() {
  // CHECK: define linkonce_odr{{.*}} void @_ZN18instantiate_member1SIiE1fEv({{.*}}
  S<int> a;
}
} // namespace instantiate_member
