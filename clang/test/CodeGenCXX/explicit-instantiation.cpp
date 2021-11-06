// RUN: %clang_cc1 -emit-llvm -triple i686-pc-linux-gnu -std=c++1y -o - %s | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-NO-OPT
// RUN: %clang_cc1 -emit-llvm -triple i686-pc-linux-gnu -std=c++1y -O3 -disable-llvm-passes -o - %s | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-OPT
// RUN: %clang_cc1 -emit-llvm -triple i686-pc-win32 -std=c++1y -o - %s | FileCheck %s --check-prefix=CHECK-MS

// This check logically is attached to 'template int S<int>::i;' below.
// CHECK: @_ZN1SIiE1iE = weak_odr global i32

// This check is logically attached to 'template int ExportedStaticLocal::f<int>()' below.
// CHECK-OPT: @_ZZN19ExportedStaticLocal1fIiEEvvE1i = linkonce_odr global

template<typename T, typename U, typename Result>
struct plus {
  Result operator()(const T& t, const U& u) const;
};

template<typename T, typename U, typename Result>
Result plus<T, U, Result>::operator()(const T& t, const U& u) const {
  return t + u;
}

// CHECK-LABEL: define weak_odr i32 @_ZNK4plusIillEclERKiRKl
template struct plus<int, long, long>;

namespace EarlyInstantiation {
  // Check that we emit definitions if we instantiate a function definition before
  // it gets explicitly instantiatied.
  template<typename T> struct S {
    constexpr int constexpr_function() { return 0; }
    auto deduced_return_type() { return 0; }
  };

  // From an implicit instantiation.
  constexpr int a = S<char>().constexpr_function();
  int b = S<char>().deduced_return_type();

  // From an explicit instantiation declaration.
  extern template struct S<int>;
  constexpr int c = S<int>().constexpr_function();
  int d = S<int>().deduced_return_type();

  // CHECK: define weak_odr i32 @_ZN18EarlyInstantiation1SIcE18constexpr_functionEv(
  // CHECK: define weak_odr i32 @_ZN18EarlyInstantiation1SIcE19deduced_return_typeEv(
  // CHECK: define weak_odr i32 @_ZN18EarlyInstantiation1SIiE18constexpr_functionEv(
  // CHECK: define weak_odr i32 @_ZN18EarlyInstantiation1SIiE19deduced_return_typeEv(
  template struct S<char>;
  template struct S<int>;

  template<typename T> constexpr int constexpr_function() { return 0; }
  template<typename T> auto deduced_return_type() { return 0; }

  // From an implicit instantiation.
  constexpr int e = constexpr_function<char>();
  int f = deduced_return_type<char>();

  // From an explicit instantiation declaration.
  extern template int constexpr_function<int>();
  extern template auto deduced_return_type<int>();
  constexpr int g = constexpr_function<int>();
  int h = deduced_return_type<int>();

  // The FIXMEs below are for PR19551.
  // CHECK: define weak_odr i32 @_ZN18EarlyInstantiation18constexpr_functionIcEEiv(
  // FIXME: define weak_odr i32 @_ZN18EarlyInstantiation19deduced_return_typeIcEEiv(
  // CHECK: define weak_odr i32 @_ZN18EarlyInstantiation18constexpr_functionIiEEiv(
  // FIXME: define weak_odr i32 @_ZN18EarlyInstantiation19deduced_return_typeIiEEiv(
  template int constexpr_function<char>();
  // FIXME template auto deduced_return_type<char>();
  template int constexpr_function<int>();
  // FIXME template auto deduced_return_type<int>();
}

namespace LateInstantiation {
  // Check that we downgrade the linkage to available_externally if we see an
  // explicit instantiation declaration after the function template is
  // instantiated.
  template<typename T> struct S { constexpr int f() { return 0; } };
  template<typename T> constexpr int f() { return 0; }

  // Trigger eager instantiation of the function definitions.
  int a, b = S<char>().f() + f<char>() + a;
  int c, d = S<int>().f() + f<int>() + a;

  // Don't allow some of those definitions to be emitted.
  extern template struct S<int>;
  extern template int f<int>();

  // Check that we declare, define, or provide an available-externally
  // definition as appropriate.
  // CHECK: define linkonce_odr i32 @_ZN17LateInstantiation1SIcE1fEv(
  // CHECK: define linkonce_odr i32 @_ZN17LateInstantiation1fIcEEiv(
  // CHECK-NO-OPT: declare i32 @_ZN17LateInstantiation1SIiE1fEv(
  // CHECK-NO-OPT: declare i32 @_ZN17LateInstantiation1fIiEEiv(
  // CHECK-OPT: define available_externally i32 @_ZN17LateInstantiation1SIiE1fEv(
  // CHECK-OPT: define available_externally i32 @_ZN17LateInstantiation1fIiEEiv(
}

namespace PR21718 {
// The linkage of a used constexpr member function can change from linkonce_odr
// to weak_odr after explicit instantiation without errors about defining the
// same function twice.
template <typename T>
struct S {
// CHECK-LABEL: define weak_odr i32 @_ZN7PR217181SIiE1fEv
  __attribute__((used)) constexpr int f() { return 0; }
};
int g() { return S<int>().f(); }
template struct S<int>;
}

namespace NestedClasses {
  // Check how explicit instantiation of an outer class affects the inner class.
  template <typename T> struct Outer {
    struct Inner {
      void f() {}
    };
  };

  // Explicit instantiation definition of Outer causes explicit instantiation
  // definition of Inner.
  template struct Outer<int>;
  // CHECK: define weak_odr void @_ZN13NestedClasses5OuterIiE5Inner1fEv
  // CHECK-MS: define weak_odr dso_local x86_thiscallcc void @"?f@Inner@?$Outer@H@NestedClasses@@QAEXXZ"

  // Explicit instantiation declaration of Outer causes explicit instantiation
  // declaration of Inner, but not in MSVC mode.
  extern template struct Outer<char>;
  auto use = &Outer<char>::Inner::f;
  // CHECK: {{declare|define available_externally}} void @_ZN13NestedClasses5OuterIcE5Inner1fEv
  // CHECK-MS: define linkonce_odr dso_local x86_thiscallcc void @"?f@Inner@?$Outer@D@NestedClasses@@QAEXXZ"
}

// Check that we emit definitions from explicit instantiations even when they
// occur prior to the definition itself.
template <typename T> struct S {
  void f();
  static void g();
  static int i;
  struct S2 {
    void h();
  };
};

// CHECK-LABEL: define weak_odr void @_ZN1SIiE1fEv
template void S<int>::f();

// CHECK-LABEL: define weak_odr void @_ZN1SIiE1gEv
template void S<int>::g();

// See the check line at the top of the file.
template int S<int>::i;

// CHECK-LABEL: define weak_odr void @_ZN1SIiE2S21hEv
template void S<int>::S2::h();

template <typename T> void S<T>::f() {}
template <typename T> void S<T>::g() {}
template <typename T> int S<T>::i;
template <typename T> void S<T>::S2::h() {}

namespace ExportedStaticLocal {
void sink(int&);
template <typename T>
inline void f() {
  static int i;
  sink(i);
}
// See the check line at the top of the file.
extern template void f<int>();
void use() {
  f<int>();
}
}

namespace DefaultedMembers {
  struct B { B(); B(const B&); ~B(); };
  template<typename T> struct A : B {
    A() = default;
    ~A() = default;
  };
  extern template struct A<int>;

  // CHECK-LABEL: define {{.*}} @_ZN16DefaultedMembers1AIiEC2Ev
  // CHECK-LABEL: define {{.*}} @_ZN16DefaultedMembers1AIiED2Ev
  A<int> ai;

  // CHECK-LABEL: define {{.*}} @_ZN16DefaultedMembers1AIiEC2ERKS1_
  A<int> ai2(ai);

  // CHECK-NOT: @_ZN16DefaultedMembers1AIcE
  template struct A<char>;
}
