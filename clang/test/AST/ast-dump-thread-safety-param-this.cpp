// RUN: %clang_cc1 -std=c++23 -verify -ast-dump %s | FileCheck %s
// RUN: %clang_cc1 -std=c++23 -fexperimental-late-parse-attributes -verify -ast-dump %s | FileCheck %s
// RUN: %clang_cc1 -std=c++11 -Wno-constexpr-not-const -verify -ast-dump %s | FileCheck --check-prefix=CXX11 %s
// expected-no-diagnostics

// A thread safety attribute on a parameter of a member function names a member
// through 'this' only if the function has an implicit object parameter.

#define REQUIRES(...) __attribute__((requires_capability(__VA_ARGS__)))
#define GUARDED_BY(x) __attribute__((guarded_by(x)))

struct __attribute__((capability("mutex"))) Mutex {};

#if __cplusplus == 201103L
// In C++11, a constexpr member function is const, also for the attributes of
// its parameters.
struct K {
  Mutex &getMu();
  const Mutex &getMu() const;
  constexpr int k(void (*cb)() REQUIRES(getMu())) { return 0; }
};
// CXX11-LABEL: CXXMethodDecl {{.*}} k 'int (void (*)()) const'
// CXX11:       RequiresCapabilityAttr
// CXX11-NEXT:  CXXMemberCallExpr {{.*}} 'const Mutex' lvalue
// CXX11-NEXT:  MemberExpr {{.*}} ->getMu
// CXX11-NEXT:  CXXThisExpr {{.*}} 'const K *' implicit this
#else
struct S {
  Mutex mu;
  void f(void (*cb)() REQUIRES(mu));
  template <class T> void tf(void (*cb)(T) REQUIRES(mu));
  static void sf(void (*cb)() REQUIRES(mu));
  void ef(this S &self [[clang::requires_capability(mu)]],
          void (*cb)() REQUIRES(mu));
  void ea([[maybe_unused]] this S &self, void (*cb)() REQUIRES(mu));
  void na([[maybe_unused]] int n, void (*cb)() REQUIRES(mu));
  // A lambda in a parameter's type has no 'this'.
  void lf(decltype([](void (*cb)() REQUIRES(mu)) {}));
  void *operator new(decltype(sizeof 0), void (*cb)() REQUIRES(mu));
  void d(int n);
  void ov(void (*cb)() REQUIRES(mu));
  static void ov(int, void (*cb)() REQUIRES(mu));
};
// An out-of-line definition is static only if its declaration is.
void S::sf(void (*cb)() REQUIRES(mu)) {}
// Nor is one whose name another member shares.
void S::ov(void (*cb)() REQUIRES(mu)) {}
void S::ov(int, void (*cb)() REQUIRES(mu)) {}
// A class in a lambda in the parameters has its own 'this'.
void S::d(int n = [] { struct L { Mutex m; int a GUARDED_BY(m); }; return 0; }()) {}

// A deduction guide is not a member function, so has no 'this'.
struct DG {
  Mutex m;
  template <class T> struct I { I(T); };
  I(void (*cb)() REQUIRES(m)) -> I<int>;
};
#endif

// CHECK-LABEL: CXXMethodDecl {{.*}} f 'void (void (*)())'
// CHECK:       RequiresCapabilityAttr
// CHECK-NEXT:  MemberExpr {{.*}} ->mu
// CHECK-NEXT:  CXXThisExpr {{.*}} 'S *' implicit this
// CHECK-LABEL: CXXMethodDecl {{.*}} tf 'void (void (*)(T))'
// CHECK:       RequiresCapabilityAttr
// CHECK-NEXT:  MemberExpr {{.*}} ->mu
// CHECK-NEXT:  CXXThisExpr {{.*}} 'S *' implicit this
// CHECK-LABEL: CXXMethodDecl {{.*}} sf 'void (void (*)())' static
// CHECK:       RequiresCapabilityAttr
// CHECK-NEXT:  DeclRefExpr {{.*}} Field {{.*}} 'mu'
// CHECK-LABEL: CXXMethodDecl {{.*}} ef 'void (S &, void (*)())'
// CHECK:       RequiresCapabilityAttr
// CHECK-NEXT:  DeclRefExpr {{.*}} Field {{.*}} 'mu'
// CHECK:       RequiresCapabilityAttr
// CHECK-NEXT:  DeclRefExpr {{.*}} Field {{.*}} 'mu'
// CHECK-LABEL: CXXMethodDecl {{.*}} ea 'void (S &, void (*)())'
// CHECK:       RequiresCapabilityAttr
// CHECK-NEXT:  DeclRefExpr {{.*}} Field {{.*}} 'mu'
// CHECK-LABEL: CXXMethodDecl {{.*}} na 'void (int, void (*)())'
// CHECK:       RequiresCapabilityAttr
// CHECK-NEXT:  MemberExpr {{.*}} ->mu
// CHECK-NEXT:  CXXThisExpr {{.*}} 'S *' implicit this
// CHECK-LABEL: CXXMethodDecl {{.*}} operator() 'void (void (*)()) const'
// CHECK:       RequiresCapabilityAttr
// CHECK-NEXT:  DeclRefExpr {{.*}} Field {{.*}} 'mu'
// CHECK-LABEL: CXXMethodDecl {{.*}} operator new
// CHECK:       RequiresCapabilityAttr
// CHECK-NEXT:  DeclRefExpr {{.*}} Field {{.*}} 'mu'
// CHECK-LABEL: CXXMethodDecl {{.*}} prev {{.*}} sf 'void (void (*)())'
// CHECK:       RequiresCapabilityAttr
// CHECK-NEXT:  DeclRefExpr {{.*}} Field {{.*}} 'mu'
// CHECK-LABEL: CXXMethodDecl {{.*}} prev {{.*}} ov 'void (void (*)())'
// CHECK:       RequiresCapabilityAttr
// CHECK-NEXT:  MemberExpr {{.*}} ->mu
// CHECK-NEXT:  CXXThisExpr {{.*}} 'S *' implicit this
// CHECK-LABEL: CXXMethodDecl {{.*}} prev {{.*}} ov 'void (int, void (*)())'
// CHECK:       RequiresCapabilityAttr
// CHECK-NEXT:  DeclRefExpr {{.*}} Field {{.*}} 'mu'
// CHECK-LABEL: FieldDecl {{.*}} a 'int'
// CHECK-NEXT:  GuardedByAttr
// CHECK-NEXT:  MemberExpr {{.*}} ->m
// CHECK-NEXT:  CXXThisExpr {{.*}} 'L *' implicit this
// CHECK-LABEL: CXXDeductionGuideDecl {{.*}} <deduction guide for I> 'auto (void (*)()) -> I<int>'
// CHECK:       RequiresCapabilityAttr
// CHECK-NEXT:  DeclRefExpr {{.*}} Field {{.*}} 'm'
