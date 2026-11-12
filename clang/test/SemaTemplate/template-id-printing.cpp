// RUN: %clang_cc1 -ast-print %s | FileCheck %s
namespace N {
  template<typename T, typename U> void f(U);
  template<int> void f();
}

void g() {
  // CHECK: N::f<int>(3.14
  N::f<int>(3.14);
  
  // CHECK: N::f<double>
  void (*fp)(int) = N::f<double>;
}


// (NNS qualified) DeclRefExpr.
namespace DRE {

template <typename T>
void foo();

void test() {
  // CHECK: DRE::foo<int>;
  DRE::foo<int>;
  // CHECK: DRE::template foo<int>;
  DRE::template foo<int>;
  // CHECK: DRE::foo<int>();
  DRE::foo<int>();
  // CHECK: DRE::template foo<int>();
  DRE::template foo<int>();
}

} // namespace DRE


// MemberExpr.
namespace ME {

struct S {
  template <typename T>
  void mem();
};

void test() {
  S s;
  // CHECK: s.mem<int>();
  s.mem<int>();
  // CHECK: s.template mem<int>();
  s.template mem<int>();
}

} // namespace ME


// UnresolvedLookupExpr.
namespace ULE {

template <typename T>
int foo();

template <typename T>
void test() {
  // CHECK: ULE::foo<T>;
  ULE::foo<T>;
  // CHECK: ULE::template foo<T>;
  ULE::template foo<T>;
}

} // namespace ULE


// UnresolvedMemberExpr.
namespace UME {

struct S {
  template <typename T>
  void mem();
};

template <typename U>
void test() {
  S s;
  // CHECK: s.mem<U>();
  s.mem<U>();
  // CHECK: s.template mem<U>();
  s.template mem<U>();
}

} // namespace UME


// DependentScopeDeclRefExpr.
namespace DSDRE {

template <typename T>
struct S;

template <typename T>
void test() {
  // CHECK: S<T>::foo;
  S<T>::foo;
  // CHECK: S<T>::template foo<>;
  S<T>::template foo<>;
  // CHECK: S<T>::template foo<T>;
  S<T>::template foo<T>;
}

} // namespace DSDRE


// DependentScopeMemberExpr.
namespace DSME {

template <typename T>
struct S;

template <typename T>
void test() {
  S<T> s;
  // CHECK: s.foo;
  s.foo;
  // CHECK: s.template foo<>;
  s.template foo<>;
  // CHECK: s.template foo<T>;
  s.template foo<T>;
}

} // namespace DSME
