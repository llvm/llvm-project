// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

void f0() &; // expected-error {{non-member function cannot have '&' qualifier}}
void f1() &&; // expected-error {{non-member function cannot have '&&' qualifier}}
void f2() const volatile &&; // expected-error {{non-member function cannot have 'const volatile &&' qualifier}}

struct X {
  void f0() &;
  void f1() &&;
  static void f2() &; // expected-error{{static member function cannot have '&' qualifier}}
  static void f3() &&; // expected-error{{static member function cannot have '&&' qualifier}}
};

typedef void func_type_lvalue() &;
typedef void func_type_rvalue() &&;

typedef func_type_lvalue *func_type_lvalue_ptr; // expected-error{{pointer to function type 'func_type_lvalue' (aka 'void () &') cannot have '&' qualifier}}
typedef func_type_rvalue *func_type_rvalue_ptr; // expected-error{{pointer to function type 'func_type_rvalue' (aka 'void () &&') cannot have '&&' qualifier}}

typedef func_type_lvalue &func_type_lvalue_ref; // expected-error{{reference to function type 'func_type_lvalue' (aka 'void () &') cannot have '&' qualifier}}
typedef func_type_rvalue &func_type_rvalue_ref; // expected-error{{reference to function type 'func_type_rvalue' (aka 'void () &&') cannot have '&&' qualifier}}

template<typename T = func_type_lvalue> struct wrap {
  typedef T val;
  typedef T *ptr;
  typedef T &ref;
};

using func_type_lvalue = wrap<>::val;
using func_type_lvalue = wrap<func_type_lvalue>::val;
using func_type_rvalue = wrap<func_type_rvalue>::val;

using func_type_lvalue_ptr = wrap<>::ptr;
using func_type_lvalue_ptr = wrap<func_type_lvalue>::ptr;
using func_type_rvalue_ptr = wrap<func_type_rvalue>::ptr;

using func_type_lvalue_ref = wrap<>::ref;
using func_type_lvalue_ref = wrap<func_type_lvalue>::ref;
using func_type_rvalue_ref = wrap<func_type_rvalue>::ref;

func_type_lvalue f2; // expected-error{{non-member function of type 'func_type_lvalue' (aka 'void () &') cannot have '&' qualifier}}
func_type_rvalue f3; // expected-error{{non-member function of type 'func_type_rvalue' (aka 'void () &&') cannot have '&&' qualifier}}

struct Y {
  func_type_lvalue f0;
  func_type_rvalue f1;
};

void (X::*mpf1)() & = &X::f0;
void (X::*mpf2)() && = &X::f1;


void (f() &&); // expected-error{{non-member function cannot have '&&' qualifier}}
