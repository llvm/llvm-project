// RUN: %clang_cc1 -triple %itanium_abi_triple -emit-llvm %s -o - | FileCheck %s
namespace foo {

// CHECK-NOT: @a = global
extern "C" int a;

// CHECK-NOT: @_ZN3foo1bE = global
extern int b;

// CHECK: @_ZN3foo1cE = {{(dso_local )?}}global
int c = 5;

// CHECK-NOT: @_ZN3foo1dE
extern "C" struct d;

// CHECK-NOT: should_not_appear
extern "C++" int should_not_appear;

// CHECK: @_ZN3foo10extern_cxxE = {{(dso_local )?}}global
extern "C++" int extern_cxx = 0;

}

// CHECK-NOT: @global_a = {{(dso_local )?}}global
extern "C" int global_a;

// CHECK: @global_b = {{(dso_local )?}}global
extern "C" int global_b = 0;

// CHECK-NOT: should_not_appear
extern "C++" int should_not_appear;

// CHECK: @extern_cxx = {{(dso_local )?}}global
extern "C++" int extern_cxx = 0;

namespace test1 {
  namespace {
    struct X {};
  }
  extern "C" {
    // CHECK: @test1_b = {{(dso_local )?}}global
    X test1_b = X();
  }
  void *use = &test1_b;
  // CHECK: @_ZN5test13useE = {{(dso_local )?}}global
}

namespace test2 {
  namespace {
    struct X {};
  }

  // CHECK: @test2_b = {{(dso_local )?}}global
  extern "C" X test2_b;
  X test2_b;
}

extern "C" {
  static int unused_var;
  static int unused_fn() { return 0; }

  __attribute__((used)) static int internal_var;
  __attribute__((used)) static int internal_fn() { return 0; }

  __attribute__((used)) static int duplicate_internal_var;
  __attribute__((used)) static int duplicate_internal_fn() { return 0; }

  namespace N {
    __attribute__((used)) static int duplicate_internal_var;
    __attribute__((used)) static int duplicate_internal_fn() { return 0; }
  }

  // CHECK: @llvm.compiler.used = appending global {{.*}} @internal_var {{.*}} @internal_fn

  // CHECK-NOT: @unused
  // CHECK-NOT: @duplicate_internal
  // CHECK: @internal_var = internal alias i32, i32* @_ZL12internal_var
  // CHECK-NOT: @unused
  // CHECK-NOT: @duplicate_internal
  // CHECK: @internal_fn = internal alias i32 (), i32 ()* @_ZL11internal_fnv
  // CHECK-NOT: @unused
  // CHECK-NOT: @duplicate_internal
}

namespace PR19411 {
  struct A { void f(); };
  extern "C" void A::f() { void g(); g(); }
  // CHECK-LABEL: @_ZN7PR194111A1fEv(
  // CHECK: call {{.*}}void @g()
}
