// RUN: %clang_cc1 -fsyntax-only -verify -fms-extensions %s -triple x86_64-pc-win32

// Multiple inheritance is involved (code segmments all disagree between the bases and derived class)
struct __declspec(code_seg("my_base")) Base1 {};
struct Base2 {};

struct D1 : Base1, Base2 {};
// expected-error@-1 {{derived class must specify the same code segment as its base classes}}
// expected-note@4 {{base class 'Base1' specified here}}

struct __declspec(code_seg("my_derived")) D2 : Base2, Base1 {};
// expected-error@-1 {{derived class must specify the same code segment as its base classes}}
// expected-error@-2 {{derived class must specify the same code segment as its base classes}}
// expected-note@5 {{base class 'Base2' specified here}}
// expected-note@4 {{base class 'Base1' specified here}}

// Multiple inheritance (code segments partially agree between the bases and the derived class)
struct __declspec(code_seg("base_class")) BaseClass1 {};
struct __declspec(code_seg("base_class")) BaseClass2 {};

struct Derived1 : BaseClass1, BaseClass2 {};
// expected-error@-1 {{derived class must specify the same code segment as its base classes}}
// expected-error@-2 {{derived class must specify the same code segment as its base classes}}
// expected-note@18 {{base class 'BaseClass1' specified here}}
// expected-note@19 {{base class 'BaseClass2' specified here}}

struct __declspec(code_seg("derived_class")) Derived2 : BaseClass2, BaseClass1 {};
// expected-error@-1 {{derived class must specify the same code segment as its base classes}}
// expected-error@-2 {{derived class must specify the same code segment as its base classes}}
// expected-note@19 {{base class 'BaseClass2' specified here}}
// expected-note@18 {{base class 'BaseClass1' specified here}}

struct __declspec(code_seg("base_class")) Derived3 : BaseClass2, BaseClass1 {}; //OK
struct __declspec(code_seg("base_class")) Derived4 : BaseClass1, BaseClass2 {}; //OK

// Multiple inheritance is involved (code segmments all agree between the bases and derived class)
struct __declspec(code_seg("foo_base")) B1 {};
struct __declspec(code_seg("foo_base")) B2 {};
struct __declspec(code_seg("foo_base")) Derived : B1, B2 {};

// virtual Inheritance is involved (code segmments all disagree between the bases and derived class)
struct __declspec(code_seg("my_one")) Base {
  virtual int barA() { return 1; } ;
};

struct __declspec(code_seg("my_two")) Derived5 : virtual Base {
  virtual int barB() { return 2; };
};
// expected-error@-3 {{derived class must specify the same code segment as its base classes}}
// expected-note@42 {{base class 'Base' specified here}}

struct __declspec(code_seg("my_three")) Derived6 : virtual Base {
  virtual int barC() { return 3; };
};
// expected-error@-3 {{derived class must specify the same code segment as its base classes}}
// expected-note@42 {{base class 'Base' specified here}}

struct __declspec(code_seg("my_four")) Derived7 : Derived5, Derived6 {};
// expected-error@-1 {{derived class must specify the same code segment as its base classes}}
// expected-error@-2 {{derived class must specify the same code segment as its base classes}}
// expected-note@46 {{base class 'Derived5' specified here}}
// expected-note@52 {{base class 'Derived6' specified here}}

// virtual Inheritance is involved (code segmments partially agree between the bases and derived class)
struct __declspec(code_seg("my_class")) BaseClass {
  virtual int barA() { return 1; } ;
};

struct __declspec(code_seg("my_class")) DerivedClass1 : virtual BaseClass { //OK
  virtual int barB() { return 2; };
};

struct __declspec(code_seg("my_class")) DerivedClass2 : virtual BaseClass { //OK
  virtual int barC() { return 3; };
};

struct __declspec(code_seg("my_derived_one")) DerivedClass3 : DerivedClass1, DerivedClass2 {};
// expected-error@-1 {{derived class must specify the same code segment as its base classes}}
// expected-error@-2 {{derived class must specify the same code segment as its base classes}}
// expected-note@69 {{base class 'DerivedClass1' specified here}}
// expected-note@73 {{base class 'DerivedClass2' specified here}}

// virtual Inheritance is involved (code segmments all agree between the bases and derived class)
struct __declspec(code_seg("foo_one")) Class {
  virtual int foo1() { return 10; } ;
};

struct __declspec(code_seg("foo_one")) Derived_One: virtual Class { //OK
  virtual int foo2() { return 20; };
};

struct __declspec(code_seg("foo_one")) Derived_Two : virtual Class { //OK
  virtual int foo3() { return 30; };
};

struct __declspec(code_seg("foo_one")) Derived_Three : Derived_One, Derived_Two {}; //OK

