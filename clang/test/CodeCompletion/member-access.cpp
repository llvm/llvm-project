struct Base1 {
  int member1;
  float member2;
};

struct Base2 {
  int member1;
  double member3;
  void memfun1(int);
};

struct Base3 : Base1, Base2 {
  void memfun1(float);
  void memfun1(double) const;
  void memfun2(int);
};

struct Derived : Base3 {
  int member4;
  int memfun3(int);
};

class Proxy {
public:
  Derived *operator->() const;
};

void test(const Proxy &p) {
  p->
}

struct Test1 {
  Base1 b;

  static void sfunc() {
    b. // expected-error {{invalid use of member 'b' in static member function}}
  }
};

struct Foo {
  void foo() const;
  static void foo(bool);
};

struct Bar {
  void foo(bool param) {
    Foo::foo(  );// unresolved member expression with an implicit base
  }
};

  // RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:29:6 %s -o - | FileCheck -check-prefix=CHECK-CC1 %s
  // CHECK-CC1: Base1 : Base1::
  // CHECK-CC1: member1 : [#int#][#Base1::#]member1
  // CHECK-CC1: member1 : [#int#][#Base2::#]member1
  // CHECK-CC1: member2 : [#float#][#Base1::#]member2
  // CHECK-CC1: member3
  // CHECK-CC1: member4
  // CHECK-CC1: memfun1 : [#void#][#Base3::#]memfun1(<#float#>)
  // CHECK-CC1: memfun1 : [#void#][#Base3::#]memfun1(<#double#>)[# const#]
  // CHECK-CC1: memfun1 (Hidden) : [#void#]Base2::memfun1(<#int#>)
  // CHECK-CC1: memfun2 : [#void#][#Base3::#]memfun2(<#int#>)
  // CHECK-CC1: memfun3 : [#int#]memfun3(<#int#>)

// Make sure this doesn't crash
// RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:36:7 %s -verify

// Make sure this also doesn't crash
// RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:47:14 %s
