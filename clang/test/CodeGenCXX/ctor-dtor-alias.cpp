// RUN: %clang_cc1 %s -triple i686-linux -emit-llvm -o - -mconstructor-aliases | FileCheck --check-prefix=NOOPT %s

// RUN: %clang_cc1 %s -triple i686-linux -emit-llvm -o - -mconstructor-aliases -O1 -disable-llvm-passes > %t
// RUN: FileCheck --check-prefix=CHECK1 --input-file=%t %s
// RUN: FileCheck --check-prefix=CHECK2 --input-file=%t %s
// RUN: FileCheck --check-prefix=CHECK3 --input-file=%t %s
// RUN: FileCheck --check-prefix=CHECK4 --input-file=%t %s
// RUN: FileCheck --check-prefix=CHECK5 --input-file=%t %s
// RUN: FileCheck --check-prefix=CHECK6 --input-file=%t %s

// RUN: %clang_cc1 %s -triple i686-pc-windows-gnu -emit-llvm -o - -mconstructor-aliases -O1 -disable-llvm-passes | FileCheck --check-prefix=COFF %s

namespace test1 {
// Test that we produce the appropriate comdats when creating aliases to
// weak_odr constructors and destructors.

// CHECK1: @_ZN5test16foobarIvEC1Ev = weak_odr unnamed_addr alias void {{.*}} @_ZN5test16foobarIvEC2Ev
// CHECK1: @_ZN5test16foobarIvED1Ev = weak_odr unnamed_addr alias void (ptr), ptr @_ZN5test16foobarIvED2Ev
// CHECK1: define weak_odr void @_ZN5test16foobarIvEC2Ev({{.*}} comdat($_ZN5test16foobarIvEC5Ev)
// CHECK1: define weak_odr void @_ZN5test16foobarIvED2Ev({{.*}} comdat($_ZN5test16foobarIvED5Ev)
// CHECK1: define weak_odr void @_ZN5test16foobarIvED0Ev({{.*}} comdat($_ZN5test16foobarIvED5Ev)
// CHECK1-NOT: comdat

// COFF doesn't support comdats with arbitrary names (C5/D5).
// COFF: define weak_odr {{.*}} void @_ZN5test16foobarIvEC2Ev({{.*}} comdat align
// COFF: define weak_odr {{.*}} void @_ZN5test16foobarIvEC1Ev({{.*}} comdat align
// COFF: define weak_odr {{.*}} void @_ZN5test16foobarIvED2Ev({{.*}} comdat align
// COFF: define weak_odr {{.*}} void @_ZN5test16foobarIvED0Ev({{.*}} comdat align

template <typename T>
struct foobar {
  foobar() {}
  virtual ~foobar() {}
};

template struct foobar<void>;
}

namespace test2 {
// test that when the destrucor is linkonce_odr we just replace every use of
// C1 with C2.

// CHECK1: define internal void @__cxx_global_var_init()
// CHECK1: call void @_ZN5test26foobarIvEC2Ev
// CHECK1: define linkonce_odr void @_ZN5test26foobarIvEC2Ev({{.*}} comdat align
void g();
template <typename T> struct foobar {
  foobar() { g(); }
};
foobar<void> x;
}

namespace test3 {
// test that instead of an internal alias we just use the other destructor
// directly.

// CHECK1: define internal void @__cxx_global_var_init.1()
// CHECK1: call i32 @__cxa_atexit{{.*}}_ZN5test312_GLOBAL__N_11AD2Ev
// CHECK1: define internal void @_ZN5test312_GLOBAL__N_11AD2Ev(
namespace {
struct A {
  ~A() {}
};

struct B : public A {};
}

B x;
}

namespace test4 {
  // Test that we don't produce aliases from B to A. We cannot because we cannot
  // guarantee that they will be present in every TU. Instead, we just call
  // A's destructor directly.

  // CHECK1: define internal void @__cxx_global_var_init.2()
  // CHECK1: call i32 @__cxa_atexit{{.*}}_ZN5test41AD2Ev
  // CHECK1: define linkonce_odr void @_ZN5test41AD2Ev({{.*}} comdat align

  // test that we don't do this optimization at -O0 so that the debugger can
  // see both destructors.
  // NOOPT: define internal void @__cxx_global_var_init.2()
  // NOOPT: call i32 @__cxa_atexit{{.*}}@_ZN5test41BD2Ev
  // NOOPT: define linkonce_odr void @_ZN5test41BD2Ev({{.*}} comdat align
  struct A {
    virtual ~A() {}
  };
  struct B : public A{
    ~B() {}
  };
  B X;
}

namespace test5 {
  // similar to test4, but with an internal B.

  // CHECK2: define internal void @__cxx_global_var_init.3()
  // CHECK2: call i32 @__cxa_atexit{{.*}}_ZN5test51AD2Ev
  // CHECK2: define linkonce_odr void @_ZN5test51AD2Ev({{.*}} comdat align
  struct A {
    virtual ~A() {}
  };
  namespace {
  struct B : public A{
    ~B() {}
  };
  }
  B X;
}

namespace test6 {
  // Test that we use ~A directly, even when ~A is not defined. The symbol for
  // ~B would have been internal and still contain a reference to ~A.
  struct A {
    virtual ~A();
  };
  namespace {
  struct B : public A {
    ~B() {}
  };
  }
  B X;
  // CHECK3: define internal void @__cxx_global_var_init.4()
  // CHECK3: call i32 @__cxa_atexit({{.*}}@_ZN5test61AD2Ev
}

namespace test7 {
  // Test that we don't produce an alias from ~B to ~A<int> (or crash figuring
  // out if we should).
  // pr17875.
  // CHECK3: define{{.*}} void @_ZN5test71BD2Ev
  template <typename> struct A {
    ~A() {}
  };
  class B : A<int> {
    ~B();
  };
  template class A<int>;
  B::~B() {}
}

namespace test8 {
  // Test that we replace ~zed with ~bar which is an alias to ~foo.
  // CHECK4: @_ZN5test83barD2Ev ={{.*}} unnamed_addr alias {{.*}} @_ZN5test83fooD2Ev
  // CHECK4: define internal void @__cxx_global_var_init.5()
  // CHECK4: call i32 @__cxa_atexit({{.*}}@_ZN5test83barD2Ev
  struct foo {
    ~foo();
  };
  foo::~foo() {}
  struct bar : public foo {
    ~bar();
  };
  bar::~bar() {}
  struct zed : public bar {};
  zed foo;
}

namespace test9 {
struct foo {
  __attribute__((stdcall)) ~foo() {
  }
};

struct bar : public foo {};

void zed() {
  // Test that we produce a call to bar's destructor. We used to call foo's, but
  // it has a different calling conversion.
  // CHECK4: call void @_ZN5test93barD2Ev
  bar ptr;
}
}

// CHECK5: @_ZTV1C = linkonce_odr unnamed_addr constant { [4 x ptr] } {{[^@]*}}@_ZTI1C, {{[^@]*}}@_ZN1CD2Ev, {{[^@]*}}@_ZN1CD0Ev]
// r194296 replaced C::~C with B::~B without emitting the later.

class A {
public:
  A(int);
  virtual ~A();
};

template <class>
class B : A {
public:
  B()
      : A(0) {
  }
  __attribute__((always_inline)) ~B() {
  }
};

extern template class B<char>;

class C : B<char> {
};

void
fn1() {
  new C;
}

namespace test10 {
// Test that if a destructor is in a comdat, we don't try to emit is as an
// alias to a base class destructor.
struct bar {
  ~bar();
};
bar::~bar() {
}
} // closing the namespace causes ~bar to be sent to CodeGen
namespace test10 {
template <typename T>
struct foo : public bar {
  ~foo();
};
template <typename T>
foo<T>::~foo() {}
template class foo<int>;
// CHECK5: define weak_odr void @_ZN6test103fooIiED2Ev({{.*}} comdat($_ZN6test103fooIiED5Ev)
}

namespace test11 {
// Test that when we don't have to worry about COMDATs we produce an alias
// from complate to base and from base to base class base.
struct bar {
  ~bar();
};
bar::~bar() {}
struct foo : public bar {
  ~foo();
};
foo::~foo() {}
// CHECK6: @_ZN6test113fooD2Ev ={{.*}} unnamed_addr alias {{.*}} @_ZN6test113barD2Ev
// CHECK6: @_ZN6test113fooD1Ev ={{.*}} unnamed_addr alias {{.*}} @_ZN6test113fooD2Ev
}

namespace test12 {
template <int>
struct foo {
  ~foo() { delete this; }
};

template class foo<1>;
// CHECK6: @_ZN6test123fooILi1EED1Ev = weak_odr unnamed_addr alias {{.*}} @_ZN6test123fooILi1EED2Ev
// CHECK6: define weak_odr void @_ZN6test123fooILi1EED2Ev({{.*}}) {{.*}} comdat($_ZN6test123fooILi1EED5Ev)
}
