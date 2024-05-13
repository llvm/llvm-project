// RUN: %clang_cc1 -std=c++11 -fsyntax-only -fsycl-is-device -verify %s

// No diagnostics
class [[clang::sycl_special_class]] class1 {
  void __init(){}
};
class __attribute__((sycl_special_class)) class2 {
  void __init(){}
};

class class3;
class [[clang::sycl_special_class]] class3 {
  void __init(){}
};

class class4;
class __attribute__((sycl_special_class)) class4 {
  void __init(){}
};

struct [[clang::sycl_special_class]] struct1 {
  void __init(){}
};
struct __attribute__((sycl_special_class)) struct2 {
  void __init(){}
};

class __attribute__((sycl_special_class)) class5;
class class5 {
  void __init(){}
};

struct __attribute__((sycl_special_class)) struct6 {
  struct6();
  bool operator==(const struct6 &);
  struct6 &operator()();
  ~struct6();
  void __init(){}
};

// Must have one and only one __init method defined
class __attribute__((sycl_special_class)) class6 { // expected-error {{types with 'sycl_special_class' attribute must have one and only one '__init' method defined}}
  class6() {}
};
class [[clang::sycl_special_class]] class7 { // expected-error {{types with 'sycl_special_class' attribute must have one and only one '__init' method defined}}
  void __init();
};

class [[clang::sycl_special_class]] class8 { // expected-error {{types with 'sycl_special_class' attribute must have one and only one '__init' method defined}}
  void __init();
  int func() {}
  void __init(int a){}
};

struct __attribute__((sycl_special_class)) struct3;
struct struct3 {}; // expected-error {{types with 'sycl_special_class' attribute must have one and only one '__init' method defined}}

// expected-error@+1{{'sycl_special_class' attribute must have one and only one '__init' method defined}}
struct __attribute__((sycl_special_class)) struct7 {
  struct7();
  bool operator==(const struct7 &);
  struct7 &operator()();
  ~struct7();
};

// Only classes
[[clang::sycl_special_class]] int var1 = 0;       // expected-warning {{'sycl_special_class' attribute only applies to classes}}
__attribute__((sycl_special_class)) int var2 = 0; // expected-warning {{'sycl_special_class' attribute only applies to classes}}

[[clang::sycl_special_class]] void foo1();       // expected-warning {{'sycl_special_class' attribute only applies to classes}}
__attribute__((sycl_special_class)) void foo2(); // expected-warning {{'sycl_special_class' attribute only applies to classes}}

// Attribute takes no arguments
class [[clang::sycl_special_class(1)]] class9{};         // expected-error {{'sycl_special_class' attribute takes no arguments}}
class __attribute__((sycl_special_class(1))) class10 {}; // expected-error {{'sycl_special_class' attribute takes no arguments}}

// __init method must be defined inside the CXXRecordDecl.
class [[clang::sycl_special_class]] class11 { // expected-error {{types with 'sycl_special_class' attribute must have one and only one '__init' method defined}}
  void __init();
};
void class11::__init(){}

class __attribute__((sycl_special_class)) class12 { // expected-error {{types with 'sycl_special_class' attribute must have one and only one '__init' method defined}}
  void __init();
};
void class12::__init(){}

struct [[clang::sycl_special_class]] struct4 { // expected-error {{types with 'sycl_special_class' attribute must have one and only one '__init' method defined}}
  void __init();
};
void struct4::__init(){}

struct __attribute__((sycl_special_class)) struct5 { // expected-error {{types with 'sycl_special_class' attribute must have one and only one '__init' method defined}}
  void __init();
};
void struct5::__init(){}
