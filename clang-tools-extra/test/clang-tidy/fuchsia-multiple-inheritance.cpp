// RUN: %check_clang_tidy %s fuchsia-multiple-inheritance %t

class Base_A {
public:
  virtual int foo() { return 0; }
};

class Base_B {
public:
  virtual int bar() { return 0; }
};

class Base_A_child : public Base_A {
public:
  virtual int baz() { return 0; }
};

class Interface_A {
public:
  virtual int foo() = 0;
};

class Interface_B {
public:
  virtual int bar() = 0;
};

class Interface_C {
public:
  virtual int blat() = 0;
};

class Interface_A_with_member {
public:
  virtual int foo() = 0;
  int val = 0;
};

class Interface_with_A_Parent : public Base_A {
public:
  virtual int baz() = 0;
};

// Inherits from multiple concrete classes.
// CHECK-MESSAGES: [[@LINE+2]]:1: warning: inheriting mulitple classes that aren't pure virtual is discouraged [fuchsia-multiple-inheritance]
// CHECK-NEXT: class Bad_Child1 : public Base_A, Base_B {};
class Bad_Child1 : public Base_A, Base_B {};

// CHECK-MESSAGES: [[@LINE+1]]:1: warning: inheriting mulitple classes that aren't pure virtual is discouraged [fuchsia-multiple-inheritance]
class Bad_Child2 : public Base_A, Interface_A_with_member {
  virtual int foo() override { return 0; }
};

// CHECK-MESSAGES: [[@LINE+2]]:1: warning: inheriting mulitple classes that aren't pure virtual is discouraged [fuchsia-multiple-inheritance]
// CHECK-NEXT: class Bad_Child3 : public Interface_with_A_Parent, Base_B {
class Bad_Child3 : public Interface_with_A_Parent, Base_B {
  virtual int baz() override { return 0; }
};

// Easy cases of single inheritance
class Simple_Child1 : public Base_A {};
class Simple_Child2 : public Interface_A {
  virtual int foo() override { return 0; }
};

// Valid uses of multiple inheritance
class Good_Child1 : public Interface_A, Interface_B {
  virtual int foo() override { return 0; }
  virtual int bar() override { return 0; }
};

class Good_Child2 : public Base_A, Interface_B {
  virtual int bar() override { return 0; }
};

class Good_Child3 : public Base_A_child, Interface_C, Interface_B {
  virtual int bar() override { return 0; }
  virtual int blat() override { return 0; }
};

struct B1 { int x; };
struct B2 { int x;};
// CHECK-MESSAGES: [[@LINE+2]]:1: warning: inheriting mulitple classes that aren't pure virtual is discouraged [fuchsia-multiple-inheritance]
// CHECK-NEXT: struct D : B1, B2 {};
struct D1 : B1, B2 {};

struct Base1 { virtual void foo() = 0; };
struct V1 : virtual Base1 {};
struct V2 : virtual Base1 {};
struct D2 : V1, V2 {};

struct Base2 { virtual void foo(); };
struct V3 : virtual Base2 {};
struct V4 : virtual Base2 {};
struct D3 : V3, V4 {};

struct Base3 {};
struct V5 : virtual Base3 { virtual void f(); };
struct V6 : virtual Base3 { virtual void g(); };
// CHECK-MESSAGES: [[@LINE+2]]:1: warning: inheriting mulitple classes that aren't pure virtual is discouraged [fuchsia-multiple-inheritance]
// CHECK-NEXT: struct D4 : V5, V6 {};
struct D4 : V5, V6 {};

struct Base4 {};
struct V7 : virtual Base4 { virtual void f() = 0; };
struct V8 : virtual Base4 { virtual void g() = 0; };
struct D5 : V7, V8 {};

struct Base5 { virtual void f() = 0; };
struct V9 : virtual Base5 { virtual void f(); };
struct V10 : virtual Base5 { virtual void g() = 0; };
struct D6 : V9, V10 {};

struct Base6 { virtual void f(); };
struct Base7 { virtual void g(); };
struct V15 : virtual Base6 { virtual void f() = 0; };
struct V16 : virtual Base7 { virtual void g() = 0; };
// CHECK-MESSAGES: [[@LINE+2]]:1: warning: inheriting mulitple classes that aren't pure virtual is discouraged [fuchsia-multiple-inheritance]
// CHECK-NEXT: struct D9 : V15, V16 {};
struct D9 : V15, V16 {};

struct Static_Base { static void foo(); };
struct V11 : virtual Static_Base {};
struct V12 : virtual Static_Base {};
struct D7 : V11, V12 {};

struct Static_Base_2 {};
struct V13 : virtual Static_Base_2 { static void f(); };
struct V14 : virtual Static_Base_2 { static void g(); };
struct D8 : V13, V14 {};

template<typename T> struct A : T {};
template<typename T> struct B : virtual T {};

template<typename> struct C {};
template<typename T> struct D : C<T> {};

// Check clang_tidy does not crash on this code.
template <class T>
struct WithTemplBase : T {
  WithTemplBase();
};

int test_no_crash() {
  auto foo = []() {};
  WithTemplBase<decltype(foo)>();
}
