// Also check that -Wglobal-constructors does the right thing. Strictly
// speaking, this is a Sema test, but this avoids test case duplication.
// RUN: %clang_cc1 -Wglobal-constructors %s -verify -triple i686-windows-msvc -fms-extensions -std=c++11
//
// RUN: %clang_cc1 %s -emit-llvm -o - -triple i686-windows-msvc -fms-extensions -std=c++11 | FileCheck %s

struct __declspec(dllimport) Single {
  void nonvirt();
  virtual void virt();
};

struct A { int a; };
struct B { int b; };
struct __declspec(dllimport) Multi : A, B {
  void nonvirt();
  virtual void virt();
};

struct __declspec(dllimport) Virtual : virtual A {
  void nonvirt();
  virtual void virt();
};

struct General;
static_assert(sizeof(void (General::*)()) == 16, "force general memptr model");
struct __declspec(dllimport) General {
  void nonvirt();
  virtual void virt();
};

auto mp_single_nv = &Single::nonvirt; // expected-warning {{global constructor}}
auto mp_multi_nv = &Multi::nonvirt; // expected-warning {{global constructor}}
auto mp_virtual_nv = &Virtual::nonvirt; // expected-warning {{global constructor}}
auto mp_general_nv = &General::nonvirt; // expected-warning {{global constructor}}

auto mp_single_v = &Single::virt;
auto mp_multi_v = &Multi::virt;
auto mp_virtual_v = &Virtual::virt;
auto mp_general_v = &General::virt;

// All of the non-virtual globals need dynamic initializers.

// CHECK: @"?mp_single_nv@@3P8Single@@AEXXZQ1@" = dso_local global ptr null, align 4
// CHECK: @"?mp_multi_nv@@3P8Multi@@AEXXZQ1@" = dso_local global { ptr, i32 } zeroinitializer, align 4
// CHECK: @"?mp_virtual_nv@@3P8Virtual@@AEXXZQ1@" = dso_local global { ptr, i32, i32 } zeroinitializer, align 4
// CHECK: @"?mp_general_nv@@3P8General@@AEXXZQ1@" = dso_local global { ptr, i32, i32, i32 } zeroinitializer, align 4

// CHECK: @"?mp_single_v@@3P8Single@@AEXXZQ1@" = dso_local global ptr @"??_9Single@@$BA@AE", align 4
// CHECK: @"?mp_multi_v@@3P8Multi@@AEXXZQ1@" = dso_local global { ptr, i32 } { ptr @"??_9Multi@@$BA@AE", i32 0 }, align 4
// CHECK: @"?mp_virtual_v@@3P8Virtual@@AEXXZQ1@" = dso_local global { ptr, i32, i32 } { ptr @"??_9Virtual@@$BA@AE", i32 0, i32 0 }, align 4
// CHECK: @"?mp_general_v@@3P8General@@AEXXZQ1@" = dso_local global { ptr, i32, i32, i32 } { ptr @"??_9General@@$BA@AE", i32 0, i32 0, i32 0 }, align 4

// CHECK: define internal void @_GLOBAL__sub_I{{.*}}() {{.*}} {
// CHECK:   call void @"??__Emp_single_nv@@YAXXZ"()
// CHECK:   call void @"??__Emp_multi_nv@@YAXXZ"()
// CHECK:   call void @"??__Emp_virtual_nv@@YAXXZ"()
// CHECK:   call void @"??__Emp_general_nv@@YAXXZ"()
// CHECK: }
