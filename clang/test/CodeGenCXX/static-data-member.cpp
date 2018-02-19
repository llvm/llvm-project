// RUN: %clang_cc1 -triple x86_64-pc-linux -std=c++14 -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin -std=c++14 -emit-llvm -o - %s \
// RUN:     | FileCheck %s --check-prefix=MACHO
// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -std=c++14 -emit-llvm -o - %s \
// RUN:     | FileCheck %s --check-prefix=MSVC
// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -std=c++17 -emit-llvm -o - %s \
// RUN:     | FileCheck %s --check-prefix=MSVC

// CHECK: @_ZN5test11A1aE = constant i32 10, align 4
// CHECK: @_ZN5test212_GLOBAL__N_11AIiE1xE = internal global i32 0, align 4
// CHECK: @_ZN5test31AIiE1xE = weak_odr global i32 0, comdat, align 4
// CHECK: @_ZGVN5test31AIiE1xE = weak_odr global i64 0, comdat($_ZN5test31AIiE1xE)
// MACHO: @_ZGVN5test31AIiE1xE = weak_odr global i64 0
// MACHO-NOT: comdat

// MSVC: @"\01?a@A@test1@@2HB" = linkonce_odr constant i32 10, comdat, align 4
// MSVC: @"\01?i@S@test1@@2HA" = external global i32
// MSVC: @"\01?x@?$A@H@?A@test2@@2HA" = internal global i32 0, align 4

// CHECK: _ZN5test51U2k0E = global i32 0
// CHECK: _ZN5test51U2k1E = global i32 0
// CHECK: _ZN5test51U2k2E = constant i32 76
// CHECK-NOT: test51U2k3E
// CHECK-NOT: test51U2k4E

// On Linux in C++14, neither of these are inline.
// CHECK: @_ZN16inline_constexpr1A10just_constE = available_externally constant i32 42
// CHECK: @_ZN16inline_constexpr1A10const_exprE = available_externally constant i32 43
//
// In MSVC, these are both implicitly inline regardless of the C++ standard
// version.
// MSVC: @"\01?just_const@A@inline_constexpr@@2HB" = linkonce_odr constant i32 42, comdat, align 4
// MSVC: @"\01?const_expr@A@inline_constexpr@@2HB" = linkonce_odr constant i32 43, comdat, align 4

// PR5564.
namespace test1 {
  struct A {
    static const int a = 10;
  };

  const int A::a;

  struct S { 
    static int i;
  };

  void f() { 
    int a = *&A::a + S::i;
  }
}

// Test that we don't use guards for initializing template static data
// members with internal linkage.
namespace test2 {
  int foo();

  namespace {
    template <class T> struct A {
      static int x;
    };

    template <class T> int A<T>::x = foo();
    template struct A<int>;
  }

  // CHECK-LABEL: define internal void @__cxx_global_var_init()
  // CHECK:      [[TMP:%.*]] = call i32 @_ZN5test23fooEv()
  // CHECK-NEXT: store i32 [[TMP]], i32* @_ZN5test212_GLOBAL__N_11AIiE1xE, align 4
  // CHECK-NEXT: ret void

  // MSVC-LABEL: define internal void @"\01??__Ex@?$A@H@?A@test2@@2HA@YAXXZ"()
  // MSVC:      [[TMP:%.*]] = call i32 @"\01?foo@test2@@YAHXZ"()
  // MSVC-NEXT: store i32 [[TMP]], i32* @"\01?x@?$A@H@?A@test2@@2HA", align 4
  // MSVC-NEXT: ret void
}

// Test that we don't use threadsafe statics when initializing
// template static data members.
namespace test3 {
  int foo();

  template <class T> struct A {
    static int x;
  };

  template <class T> int A<T>::x = foo();
  template struct A<int>;

  // CHECK-LABEL: define internal void @__cxx_global_var_init.1() {{.*}} comdat($_ZN5test31AIiE1xE)
  // MACHO-LABEL: define internal void @__cxx_global_var_init.1()
  // MACHO-NOT: comdat
  // CHECK:      [[GUARDBYTE:%.*]] = load i8, i8* bitcast (i64* @_ZGVN5test31AIiE1xE to i8*)
  // CHECK-NEXT: [[UNINITIALIZED:%.*]] = icmp eq i8 [[GUARDBYTE]], 0
  // CHECK-NEXT: br i1 [[UNINITIALIZED]]
  // CHECK:      [[TMP:%.*]] = call i32 @_ZN5test33fooEv()
  // CHECK-NEXT: store i32 [[TMP]], i32* @_ZN5test31AIiE1xE, align 4
  // CHECK-NEXT: store i64 1, i64* @_ZGVN5test31AIiE1xE
  // CHECK-NEXT: br label
  // CHECK:      ret void
}

// Test that we can fold member lookup expressions which resolve to static data
// members.
namespace test4 {
  struct A {
    static const int n = 76;
  };

  int f(A *a) {
    // CHECK-LABEL: define i32 @_ZN5test41fEPNS_1AE
    // CHECK: ret i32 76
    return a->n;
  }
}

// Test that static data members in unions behave properly.
namespace test5 {
  union U {
    static int k0;
    static const int k1;
    static const int k2 = 76;
    static const int k3;
    static const int k4 = 81;
  };
  int U::k0;
  const int U::k1 = (k0 = 9, 42);
  const int U::k2;

  // CHECK: store i32 9, i32* @_ZN5test51U2k0E
  // CHECK: store i32 {{.*}}, i32* @_ZN5test51U2k1E
  // CHECK-NOT: store {{.*}} i32* @_ZN5test51U2k2E
}

// Test that MSVC mode static constexpr data members are always inline, even pre
// C++17.
namespace inline_constexpr {
struct A {
  static const int just_const = 42;
  static constexpr int const_expr = 43;
};
int useit() { return *&A::just_const + *&A::const_expr; }
}
