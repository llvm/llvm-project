// RUN: %clang_cc1 -emit-llvm -triple i686-pc-win32 -fms-extensions -verify -o - %s | FileCheck %s
// expected-no-diagnostics
// The Microsoft document says: "When this attribute is applied to a class,
// all member functions of the class and nested classes - this includes
// compiler-generated special member functions - are put in the specified segment."
// But the MS compiler does not always follow that.  A bug has been reported:
// see https://reviews.llvm.org/D22931, the Microsoft feedback page is no
// longer available.
// The MS compiler will apply a declspec from the parent class if there is no
// #pragma code_seg active at the class definition.  If there is an active
// code_seg that is used instead.

// No active code_seg

struct __declspec(code_seg("foo_outer")) Foo1 {
  struct Inner {
    void bar1();
    static void bar2();
  };
};
void Foo1::Inner::bar1() {}
void Foo1::Inner::bar2() {}

//CHECK: define {{.*}}bar1@Inner@Foo1{{.*}} section "foo_outer"
//CHECK: define {{.*}}bar2@Inner@Foo1{{.*}} section "foo_outer"

struct __declspec(code_seg("foo_outer")) Foo2 {
  struct __declspec(code_seg("foo_inner")) Inner {
    void bar1();
    static void bar2();
  };
};
void Foo2::Inner::bar1() {}
void Foo2::Inner::bar2() {}

//CHECK: define {{.*}}bar1@Inner@Foo2{{.*}} section "foo_inner"
//CHECK: define {{.*}}bar2@Inner@Foo2{{.*}} section "foo_inner"

#pragma code_seg(push, "otherseg")
struct __declspec(code_seg("foo_outer")) Foo3 {
  struct Inner {
    void bar1();
    static void bar2();
  };
};
void Foo3::Inner::bar1() {}
void Foo3::Inner::bar2() {}

//CHECK: define {{.*}}bar1@Inner@Foo3{{.*}} section "otherseg"
//CHECK: define {{.*}}bar2@Inner@Foo3{{.*}} section "otherseg"

struct __declspec(code_seg("foo_outer")) Foo4 {
  struct __declspec(code_seg("foo_inner")) Inner {
    void bar1();
    static void bar2();
  };
};
void Foo4::Inner::bar1() {}
void Foo4::Inner::bar2() {}

//CHECK: define {{.*}}bar1@Inner@Foo4{{.*}} section "foo_inner"
//CHECK: define {{.*}}bar2@Inner@Foo4{{.*}} section "foo_inner"

#pragma code_seg(pop)
// Back to no active pragma
struct __declspec(code_seg("foo_outer")) Foo5 {
  struct Inner {
    void bar1();
    static void bar2();
    struct __declspec(code_seg("inner1_seg")) Inner1 {
      struct Inner2 {
        void bar1();
        static void bar2();
      };
    };
  };
};
void Foo5::Inner::bar1() {}
void Foo5::Inner::bar2() {}
void Foo5::Inner::Inner1::Inner2::bar1() {}
void Foo5::Inner::Inner1::Inner2::bar2() {}

//CHECK: define {{.*}}bar1@Inner@Foo5{{.*}} section "foo_outer"
//CHECK: define {{.*}}bar2@Inner@Foo5{{.*}} section "foo_outer"
//CHECK: define {{.*}}bar1@Inner2@Inner1@Inner@Foo5{{.*}} section "inner1_seg"
//CHECK: define {{.*}}bar2@Inner2@Inner1@Inner@Foo5{{.*}} section "inner1_seg"

