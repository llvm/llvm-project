// RUN: %clang_cc1 -fsyntax-only -verify -fms-extensions %s -triple x86_64-pc-win32

struct __declspec(code_seg("my_one")) FooOne {
  int barC();
};

struct FooTwo {
  int __declspec(code_seg("my_three")) barD();
  int barE();
};
int __declspec(code_seg("my_four")) FooOne::barC() { return 10; }
// expected-warning@-1 {{section does not match previous declaration}}
// expected-note@3{{previous attribute is here}}
int __declspec(code_seg("my_five")) FooTwo::barD() { return 20; }
// expected-warning@-1 {{section does not match previous declaration}}
// expected-note@8 {{previous attribute is here}}
int __declspec(code_seg("my_six")) FooTwo::barE() { return 30; }
// expected-warning@-1 {{section does not match previous declaration}}
// expected-note@9 {{previous declaration is here}}

// Microsoft docs say:
// If a base-class has a code_seg attribute, derived classes must have the
// same attribute.
struct __declspec(code_seg("my_base")) Base1 {};
struct Base2 {};

struct D1 : Base1 {};
//expected-error@-1 {{derived class must specify the same code segment as its base classes}}
// expected-note@24 {{base class 'Base1' specified here}}
struct __declspec(code_seg("my_derived")) D2 : Base1 {};
// expected-error@-1 {{derived class must specify the same code segment as its base classes}}
// expected-note@24 {{base class 'Base1' specified here}}
struct __declspec(code_seg("my_derived")) D3 : Base2 {};
// expected-error@-1 {{derived class must specify the same code segment as its base classes}}
// expected-note@25 {{base class 'Base2' specified here}}

template <typename T> struct __declspec(code_seg("my_base")) MB : T { };
template <typename T> struct __declspec(code_seg("my_derived")) MD : T { };
MB<Base1> mb1; // ok
MB<Base2> mb2;
// expected-error@37 {{derived class must specify the same code segment as its base classes}}
// expected-note@-2 {{in instantiation of template class}}
// expected-note@25  {{base class 'Base2' specified here}}
MD<Base1> md1;
// expected-error@38 {{derived class must specify the same code segment as its base classes}}
// expected-note@-2 {{in instantiation of template class}}
// expected-note@24 {{base class 'Base1' specified here}}
MD<Base2> md2;
// expected-error@38 {{derived class must specify the same code segment as its base classes}}
// expected-note@-2 {{in instantiation of template class}}
// expected-note@25 {{base class 'Base2' specified here}}

// Virtual overrides must have the same code_seg.
struct __declspec(code_seg("my_one")) Base3 {
  virtual int barA() { return 1; }
  virtual int __declspec(code_seg("my_two")) barB() { return 2; }
};
struct __declspec(code_seg("my_one")) Derived3 : Base3 {
  int barA() { return 4; } // ok
  int barB() { return 6; }
  // expected-error@-1 {{overriding virtual function must specify the same code segment as its overridden function}}
  // expected-note@56 {{previous declaration is here}}
};

struct Base4 {
  virtual int __declspec(code_seg("my_one")) barA() {return 1;}
  virtual int barB() { return 2;}
};
struct Derived4 : Base4 {
  virtual int barA() {return 1;}
  // expected-error@-1 {{overriding virtual function must specify the same code segment as its overridden function}}
  // expected-note@66 {{previous declaration is here}}
  virtual int __declspec(code_seg("my_two")) barB() {return 1;}
  // expected-error@-1 {{overriding virtual function must specify the same code segment as its overridden function}}
  // expected-note@67 {{previous declaration is here}}
};

// MS gives an error when different code segments are used but a warning when a duplicate is used

// Function
int __declspec(code_seg("foo")) __declspec(code_seg("foo")) bar1() { return 1; }
// expected-warning@-1 {{duplicate code segment specifiers}}
int __declspec(code_seg("foo")) __declspec(code_seg("bar")) bar2() { return 1; }
// expected-error@-1 {{conflicting code segment specifiers}}

// Class
struct __declspec(code_seg("foo")) __declspec(code_seg("foo")) Foo {
  // expected-warning@-1 {{duplicate code segment specifiers}}
  int bar3() {return 0;}
};
struct __declspec(code_seg("foo")) __declspec(code_seg("bar")) FooSix {
  // expected-error@-1 {{conflicting code segment specifiers}}
  int bar3() {return 0;}
};

//Class Members
struct FooThree {
  int __declspec(code_seg("foo")) __declspec(code_seg("foo")) bar1() { return 1; }
  // expected-warning@-1 {{duplicate code segment specifiers}}
  int __declspec(code_seg("foo")) __declspec(code_seg("bar")) bar2() { return 1; }
  // expected-error@-1 {{conflicting code segment specifiers}}
  int bar8();
  int bar9() { return 9; }
};
