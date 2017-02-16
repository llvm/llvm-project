// Clear and create directories
// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: mkdir %t/cache
// RUN: mkdir %t/Inputs

// Build first header file
// RUN: echo "#define FIRST" >> %t/Inputs/first.h
// RUN: cat %s               >> %t/Inputs/first.h

// Build second header file
// RUN: echo "#define SECOND" >> %t/Inputs/second.h
// RUN: cat %s                >> %t/Inputs/second.h

// Build module map file
// RUN: echo "module first {"           >> %t/Inputs/module.map
// RUN: echo "    header \"first.h\""   >> %t/Inputs/module.map
// RUN: echo "}"                        >> %t/Inputs/module.map
// RUN: echo "module second {"          >> %t/Inputs/module.map
// RUN: echo "    header \"second.h\""  >> %t/Inputs/module.map
// RUN: echo "}"                        >> %t/Inputs/module.map

// Run test
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/cache -x c++ -I%t/Inputs -verify %s -std=c++11

#if !defined(FIRST) && !defined(SECOND)
#include "first.h"
#include "second.h"
#endif

#if defined(FIRST)
struct S1 {
  public:
};
#elif defined(SECOND)
struct S1 {
  private:
};
#else
S1 s1;
// expected-error@first.h:* {{'S1' has different definitions in different modules; first difference is definition in module 'first' found public access specifier}}
// expected-note@second.h:* {{but in 'second' found private access specifier}}
#endif

#if defined(FIRST)
struct S2Friend2 {};
struct S2 {
  friend S2Friend2;
};
#elif defined(SECOND)
struct S2Friend1 {};
struct S2 {
  friend S2Friend1;
};
#else
S2 s2;
// expected-error@first.h:* {{'S2' has different definitions in different modules; first difference is definition in module 'first' found friend 'S2Friend2'}}
// expected-note@second.h:* {{but in 'second' found other friend 'S2Friend1'}}
#endif

#if defined(FIRST)
template<class>
struct S3Template {};
struct S3 {
  friend S3Template<int>;
};
#elif defined(SECOND)
template<class>
struct S3Template {};
struct S3 {
  friend S3Template<double>;
};
#else
S3 s3;
// expected-error@first.h:* {{'S3' has different definitions in different modules; first difference is definition in module 'first' found friend 'S3Template<int>'}}
// expected-note@second.h:* {{but in 'second' found other friend 'S3Template<double>'}}
#endif

#if defined(FIRST)
struct S4 {
  static_assert(1 == 1, "First");
};
#elif defined(SECOND)
struct S4 {
  static_assert(1 == 1, "Second");
};
#else
S4 s4;
// expected-error@first.h:* {{'S4' has different definitions in different modules; first difference is definition in module 'first' found static assert with message}}
// expected-note@second.h:* {{but in 'second' found static assert with different message}}
#endif

#if defined(FIRST)
struct S5 {
  static_assert(1 == 1, "Message");
};
#elif defined(SECOND)
struct S5 {
  static_assert(2 == 2, "Message");
};
#else
S5 s5;
// expected-error@first.h:* {{'S5' has different definitions in different modules; first difference is definition in module 'first' found static assert with condition}}
// expected-note@second.h:* {{but in 'second' found static assert with different condition}}
#endif

#if defined(FIRST)
struct S6 {
  int First();
};
#elif defined(SECOND)
struct S6 {
  int Second();
};
#else
S6 s6;
// expected-error@second.h:* {{'S6::Second' from module 'second' is not present in definition of 'S6' in module 'first'}}
// expected-note@first.h:* {{definition has no member 'Second'}}
#endif

#if defined(FIRST)
struct S7 {
  double foo();
};
#elif defined(SECOND)
struct S7 {
  int foo();
};
#else
S7 s7;
// expected-error@second.h:* {{'S7::foo' from module 'second' is not present in definition of 'S7' in module 'first'}}
// expected-note@first.h:* {{declaration of 'foo' does not match}}
#endif

#if defined(FIRST)
struct S8 {
  void foo();
};
#elif defined(SECOND)
struct S8 {
  void foo() {}
};
#else
S8 s8;
// expected-error@first.h:* {{'S8' has different definitions in different modules; definition in module 'first' is here}}
// expected-note@second.h:* {{definition in module 'second' is here}}
#endif

#if defined(FIRST)
struct S9 {
  void foo() { int y = 5; }
};
#elif defined(SECOND)
struct S9 {
  void foo() { int x = 5; }
};
#else
S9 s9;
// expected-error@first.h:* {{'S9' has different definitions in different modules; definition in module 'first' is here}}
// expected-note@second.h:* {{definition in module 'second' is here}}
#endif

#if defined(FIRST)
struct S10 {
  struct {
    int x;
  } a;
};
#elif defined(SECOND)
struct S10 {
  struct {
    int x;
    int y;
  } a;
};
#else
S10 s10;
// expected-error-re@second.h:* {{'S10::(anonymous struct)::y' from module 'second' is not present in definition of 'S10::(anonymous struct at {{.*}}first.h:{{[0-9]*}}:{{[0-9]*}})' in module 'first'}}
// expected-note@first.h:* {{definition has no member 'y'}}

// expected-error@first.h:* {{'S10' has different definitions in different modules; definition in module 'first' is here}}
// expected-note@second.h:* {{definition in module 'second' is here}}
#endif

#if defined(FIRST)
struct S11 {
  void foo() { int y = sizeof(int); }
};
#elif defined(SECOND)
struct S11 {
  void foo() { int y = sizeof(double); }
};
#else
S11 s11;
// expected-error@first.h:* {{'S11' has different definitions in different modules; definition in module 'first' is here}}
// expected-note@second.h:* {{definition in module 'second' is here}}
#endif

#if defined(FIRST)
struct S12 {
  int x = sizeof(x);
  int y = sizeof(x);
};
#elif defined(SECOND)
struct S12 {
  int x = sizeof(x);
  int y = sizeof(y);
};
#else
S12 s12;
// expected-error@first.h:* {{'S12' has different definitions in different modules; definition in module 'first' is here}}
// expected-note@second.h:* {{definition in module 'second' is here}}
#endif

#if defined(FIRST)
struct S13 {
  template <typename A> void foo();
};
#elif defined(SECOND)
struct S13 {
  template <typename B> void foo();
};
#else
S13 s13;
// expected-error@first.h:* {{'S13' has different definitions in different modules; definition in module 'first' is here}}
// expected-note@second.h:* {{definition in module 'second' is here}}
#endif

#if defined(FIRST)
struct S14 {
  template <typename A, typename B> void foo();
};
#elif defined(SECOND)
struct S14 {
  template <typename A> void foo();
};
#else
S14 s14;
// expected-error@second.h:* {{'S14::foo' from module 'second' is not present in definition of 'S14' in module 'first'}}
// expected-note@first.h:* {{declaration of 'foo' does not match}}
#endif

#if defined(FIRST)
template <typename T>
struct S15 : T {
  void foo() {
    int x = __builtin_offsetof(T, first);
  }
};
#elif defined(SECOND)
template <typename T>
struct S15 : T {
  void foo() {
    int x = __builtin_offsetof(T, second);
  }
};
#else
template <typename T>
void Test15() {
  S15<T> s15;
// expected-error@first.h:* {{'S15' has different definitions in different modules; definition in module 'first' is here}}
// expected-note@second.h:* {{definition in module 'second' is here}}
}
#endif

#if defined(FIRST)
struct S16 {
  template <template<int = 0> class Y>
  void foo() {
    Y<> y;
  }
};
#elif defined(SECOND)
struct S16 {
  template <template<int = 1> class Y>
  void foo() {
    Y<> y;
  }
};
#else
void TestS16() {
  S16 s16;
}
// expected-error@first.h:* {{'S16' has different definitions in different modules; definition in module 'first' is here}}
// expected-note@second.h:* {{definition in module 'second' is here}}
#endif

#if defined(FIRST)
struct S17 {
  template <template <typename> class T>
  static int foo(int a = 1);
  template <template <typename> class T, template <typename> class U>
  using Q_type = T<int>;
};
#elif defined(SECOND)
struct S17 {
  template <template <typename> class T>
  static int foo(int a = 1);
  template <template <typename> class T, template <typename> class U>
  using Q_type = U<int>;
};
#else
S17 s17;
// expected-error@second.h:* {{'S17::Q_type' from module 'second' is not present in definition of 'S17' in module 'first'}}
// expected-note@first.h:* {{declaration of 'Q_type' does not match}}
#endif

#if defined(FIRST)
struct S18 {
  enum E { X1 };
};
#elif defined(SECOND)
struct S18 {
  enum X { X1 };
};
#else
S18 s18;
// expected-error@second.h:* {{'S18::X' from module 'second' is not present in definition of 'S18' in module 'first'}}
// expected-note@first.h:* {{definition has no member 'X'}}
#endif

#if defined(FIRST)
struct S19 {
  enum E { X1 };
};
#elif defined(SECOND)
struct S19 {
  enum E { X1, X2 };
};
#else
S19 s19;
// expected-error@first.h:* {{'S19' has different definitions in different modules; first difference is definition in module 'first' found enum 'E' has 1 element}}
// expected-note@second.h:* {{but in 'second' found enum 'E' has 2 elements}}
// expected-error@second.h:* {{'S19::E::X2' from module 'second' is not present in definition of 'S19::E' in module 'first'}}
// expected-note@first.h:* {{definition has no member 'X2'}}
#endif

#if defined(FIRST)
struct S20 {
  enum E { X1 = 1 };
};
#elif defined(SECOND)
struct S20 {
  enum E { X1 = 5};
};
#else
S20 s20;
// expected-error@first.h:* {{'S20' has different definitions in different modules; first difference is definition in module 'first' found element 'X1' in enum 'E' with initializer}}
// expected-note@second.h:* {{but in 'second' found element 'X1' in enum 'E' with different initializer}}
#endif

#if defined(FIRST)
struct S21 {
  void foo() {
    label:
    ;
  }
};
#elif defined(SECOND)
struct S21 {
  void foo() {
    ;
  }
};
#else
S21 s21;
// expected-error@first.h:* {{'S21' has different definitions in different modules; definition in module 'first' is here}}
// expected-note@second.h:* {{definition in module 'second' is here}}
#endif

#if defined(FIRST)
struct S22 {
  void foo() {
    label_first:
    ;
  }
};
#elif defined(SECOND)
struct S22 {
  void foo() {
    label_second:
    ;
  }
};
#else
S22 s22;
// expected-error@first.h:* {{'S22' has different definitions in different modules; definition in module 'first' is here}}
// expected-note@second.h:* {{definition in module 'second' is here}}
#endif

#if defined(FIRST)
struct S23 {
  typedef int a;
  typedef char b;
};
#elif defined(SECOND)
struct S23 {
  typedef char a;
  typedef int b;
};
#else
S23 s23;
// expected-error@second.h:* {{'S23::a' from module 'second' is not present in definition of 'S23' in module 'first'}}
// expected-note@first.h:* {{declaration of 'a' does not match}}
// expected-error@second.h:* {{'S23::b' from module 'second' is not present in definition of 'S23' in module 'first'}}
// expected-note@first.h:* {{declaration of 'b' does not match}}
#endif

#if defined(FIRST)
struct S24 {
  inline int foo();
};
#elif defined(SECOND)
struct S24 {
  int foo();
};
#else
S24 s24;
// expected-error@first.h:* {{'S24' has different definitions in different modules; first difference is definition in module 'first' found method 'foo' is inline}}
// expected-note@second.h:* {{but in 'second' found method 'foo' is not inline}}
#endif

#if defined(FIRST)
struct S25 {
  int x;
  S25() : x(5) {}
};
#elif defined(SECOND)
struct S25 {
  int x;
  S25() {}
};
#else
S25 s25;
// expected-error@first.h:* {{'S25' has different definitions in different modules; definition in module 'first' is here}}
// expected-note@second.h:* {{definition in module 'second' is here}}
#endif

#if defined(FIRST)
struct S26 {
  int x;
  S26() : x(5) {}
};
#elif defined(SECOND)
struct S26 {
  int x;
  S26() : x(2) {}
};
#else
S26 s26;
// expected-error@first.h:* {{'S26' has different definitions in different modules; definition in module 'first' is here}}
// expected-note@second.h:* {{definition in module 'second' is here}}
#endif

#if defined(FIRST)
struct S27 {
  explicit S27(int) {}
  S27() {}
};
#elif defined(SECOND)
struct S27 {
  S27(int) {}
  S27() {}
};
#else
S27 s27;
// expected-error@first.h:* {{'S27' has different definitions in different modules; definition in module 'first' is here}}
// expected-note@second.h:* {{definition in module 'second' is here}}
#endif

#if defined(FIRST) || defined(SECOND)
struct Base1 {
  Base1();
  Base1(int);
  Base1(double);
};

struct Base2 {
  Base2();
  Base2(int);
  Base2(double);
};
#endif

#if defined(FIRST)
struct S28 : public Base1 {};
#elif defined(SECOND)
struct S28 : public Base2 {};
#else
S28 s28;
// expected-error@first.h:* {{'S28' has different definitions in different modules; definition in module 'first' is here}}
// expected-note@second.h:* {{definition in module 'second' is here}}
#endif

#if defined(FIRST)
struct S29 : virtual Base1 {};
#elif defined(SECOND)
struct S29 : virtual Base2 {};
#else
S29 s29;
// expected-error@first.h:* {{'S29' has different definitions in different modules; definition in module 'first' is here}}
// expected-note@second.h:* {{definition in module 'second' is here}}
#endif

#if defined(FIRST)
struct S30 : public Base1 {
  S30() : Base1(1) {}
};
#elif defined(SECOND)
struct S30 : public Base1 {
  S30() : Base1(1.0) {}
};
#else
S30 s30;
// expected-error@first.h:* {{'S30' has different definitions in different modules; definition in module 'first' is here}}
// expected-note@second.h:* {{definition in module 'second' is here}}
#endif

#if defined(FIRST)
struct S31 : virtual Base1 {
  S31() : Base1(1) {}
};
#elif defined(SECOND)
struct S31 : virtual Base1 {
  S31() : Base1(1.0) {}
};
#else
S31 s31;
// expected-error@first.h:* {{'S31' has different definitions in different modules; definition in module 'first' is here}}
// expected-note@second.h:* {{definition in module 'second' is here}}
#endif

#if defined(FIRST)
struct S32 : public Base1, Base2 {
  S32() : Base1(1), Base2(1.0) {}
};
#elif defined(SECOND)
struct S32 : public Base2, Base1 {
  S32() : Base2(1), Base1(1.0) {}
};
#else
S32 s32;
// expected-error@first.h:* {{'S32' has different definitions in different modules; definition in module 'first' is here}}
// expected-note@second.h:* {{definition in module 'second' is here}}
#endif

#if defined(FIRST)
struct S33 {
  S33() : S33(5) {}
  S33(int) {int a;}
};
#elif defined(SECOND)
struct S33 {
  S33() : S33(5) {}
  S33(int) {}
};
#else
S33 s33;
// expected-error@first.h:* {{'S33' has different definitions in different modules; definition in module 'first' is here}}
// expected-note@second.h:* {{definition in module 'second' is here}}
#endif

#if defined(FIRST)
struct S34 {
  operator bool();
};
#elif defined(SECOND)
struct S34 {
  operator int();
};
#else
S34 s34;
// expected-error@second.h:* {{'S34::operator int' from module 'second' is not present in definition of 'S34' in module 'first'}}
// expected-note@first.h:* {{definition has no member 'operator int'}}
#endif

#if defined(FIRST)
struct S35 {
  explicit operator bool();
};
#elif defined(SECOND)
struct S35 {
  operator bool();
};
#else
S35 s35;
// expected-error@first.h:* {{'S35' has different definitions in different modules; definition in module 'first' is here}}
// expected-note@second.h:* {{definition in module 'second' is here}}
#endif

#if defined(FIRST)
struct S36 {
  int x : 3;
};
#elif defined(SECOND)
struct S36 {
  int x : 4;
};
#else
S36 s36;
// expected-error@first.h:* {{'S36' has different definitions in different modules; first difference is definition in module 'first' found bitfield 'x'}}
// expected-note@second.h:* {{but in 'second' found bitfield 'x'}}
#endif

#if defined(FIRST)
struct S37 {
  mutable int x;
  int y;
};
#elif defined(SECOND)
struct S37 {
  int x;
  mutable int y;
};
#else
S37 s37;
// expected-error@first.h:* {{'S37' has different definitions in different modules; first difference is definition in module 'first' found mutable 'x'}}
// expected-note@second.h:* {{but in 'second' found non-mutable 'x'}}
#endif

#if defined(FIRST)
template <class X>
struct S38 { };
#elif defined(SECOND)
template <class Y>
struct S38 { };
#else
S38<int> s38;
// expected-error@first.h:* {{'S38' has different definitions in different modules; definition in module 'first' is here}}
// expected-note@second.h:* {{definition in module 'second' is here}}
#endif

#if defined(FIRST)
template <class X = int>
struct S39 { X x; };
#elif defined(SECOND)
template <class X = double>
struct S39 { X x; };
#else
S39<> s39;
// expected-error@first.h:* {{'S39' has different definitions in different modules; definition in module 'first' is here}}
// expected-note@second.h:* {{definition in module 'second' is here}}
#endif

#if defined(FIRST)
template <int X = 5>
struct S40 { int x = X; };
#elif defined(SECOND)
template <int X = 7>
struct S40 { int x = X; };
#else
S40<> s40;
// expected-error@first.h:* {{'S40' has different definitions in different modules; definition in module 'first' is here}}
// expected-note@second.h:* {{definition in module 'second' is here}}

#endif

#if defined(FIRST)
template <int> class T41a{};
template <template<int> class T = T41a>
struct S41 {};
#elif defined(SECOND)
template <int> class T41b{};
template <template<int> class T = T41b>
struct S41 {};
#else
using ::S41;
// expected-error@first.h:* {{'S41' has different definitions in different modules; definition in module 'first' is here}}
// expected-note@second.h:* {{definition in module 'second' is here}}
#endif

#if defined(FIRST)
struct S42 {
  void foo() const {}
  void bar() {}
};
#elif defined(SECOND)
struct S42 {
  void foo() {}
  void bar() const {}
};
#else
S42 s42;
// expected-error@second.h:* {{'S42::bar' from module 'second' is not present in definition of 'S42' in module 'first'}}
// expected-note@first.h:* {{declaration of 'bar' does not match}}
// expected-error@second.h:* {{'S42::foo' from module 'second' is not present in definition of 'S42' in module 'first'}}
// expected-note@first.h:* {{declaration of 'foo' does not match}}
#endif

#if defined(FIRST)
struct S43 {
  static constexpr int x = 1;
  int y = 1;
};
#elif defined(SECOND)
struct S43 {
  int x = 1;
  static constexpr int y = 1;
};
#else
S43 s43;
// expected-error@second.h:* {{'S43::x' from module 'second' is not present in definition of 'S43' in module 'first'}}
// expected-note@first.h:* {{declaration of 'x' does not match}}
// expected-error@second.h:* {{'S43::y' from module 'second' is not present in definition of 'S43' in module 'first'}}
// expected-note@first.h:* {{declaration of 'y' does not match}}
//#endif
#endif

#if defined(FIRST)
void f44();
struct S44 {
  friend void f44();
};
#elif defined(SECOND)
void g44();
struct S44 {
  friend void g44();
};
#else
S44 s44;
// expected-error@first.h:* {{'S44' has different definitions in different modules; first difference is definition in module 'first' found friend 'f44'}}
// expected-note@second.h:* {{but in 'second' found other friend 'g44'}}
#endif

#if defined(FIRST)
struct S45 { int n : 1; };
#elif defined(SECOND)
struct S45 { int n = 1; };
#else
S45 s45;
// expected-error@first.h:* {{'S45' has different definitions in different modules; first difference is definition in module 'first' found bitfield 'n'}}
// expected-note@second.h:* {{but in 'second' found field 'n'}}
#endif

#if defined(FIRST)
struct S46 {
  int operator+(int) { return 0; }
};
#elif defined(SECOND)
struct S46 {
  int operator-(int) { return 0; }
};
#else
S46 s46;
// expected-error@second.h:* {{'S46::operator-' from module 'second' is not present in definition of 'S46' in module 'first'}}
// expected-note@first.h:* {{definition has no member 'operator-'}}
#endif

#if defined(FIRST)
template <typename T>
struct S47 {
  int foo(int);
  float foo(float);
  int bar(int);
  float bar(float);
  int x = foo(T());
};
#elif defined(SECOND)
template <typename T>
struct S47 {
  int foo(int);
  float foo(float);
  int bar(int);
  float bar(float);
  int x = bar(T());
};
#else
template <typename T>
using S48 = S47<T>;
// expected-error@first.h:* {{'S47' has different definitions in different modules; definition in module 'first' is here}}
// expected-note@second.h:* {{definition in module 'second' is here}}
#endif

#if defined(FIRST)
template <typename T>
struct S49 {
  int operator+(int);
  float operator+(float);
  int operator-(int);
  float operator-(float);
  int x = S49() + T();
};
#elif defined(SECOND)
template <typename T>
struct S49 {
  int operator+(int);
  float operator+(float);
  int operator-(int);
  float operator-(float);
  int x = S49() - T();
};
#else
template <typename T>
using S50 = S49<T>;
// expected-error@first.h:* {{'S49' has different definitions in different modules; definition in module 'first' is here}}
// expected-note@second.h:* {{definition in module 'second' is here}}
#endif

#if defined(FIRST)
namespace A51 {
  void foo();
}
struct S51 {
  S51() {
    A51::foo();
  }
};
#elif defined(SECOND)
namespace B51 {
  void foo();
}
struct S51 {
  S51() {
    B51::foo();
  }
};
#else
S51 s51;
// expected-error@first.h:* {{'S51' has different definitions in different modules; definition in module 'first' is here}}
// expected-note@second.h:* {{definition in module 'second' is here}}
#endif

#if defined(FIRST)
namespace N52 {
  void foo();
}
struct S52 {
  S52() {
    N52::foo();
  }
};
#elif defined(SECOND)
namespace N52 {
  void foo();
}
struct S52 {
  S52() {
    ::N52::foo();
  }
};
#else
S52 s52;
// expected-error@first.h:* {{'S52' has different definitions in different modules; definition in module 'first' is here}}
// expected-note@second.h:* {{definition in module 'second' is here}}
#endif

#if defined(FIRST)
namespace N53 {
  struct foo {
    static int bar();
  };
  using A = foo;
}
struct S53 {
  S53() {
    N53::A::bar();
  }
};
#elif defined(SECOND)
namespace N53 {
  struct foo {
    static int bar();
  };
  using B = foo;
}
struct S53 {
  S53() {
    N53::B::bar();
  }
};
#else
S53 s53;
// expected-error@first.h:* {{'S53' has different definitions in different modules; definition in module 'first' is here}}
// expected-note@second.h:* {{definition in module 'second' is here}}
#endif

#if defined(FIRST)
namespace N54 {
namespace A {
void foo();
}
namespace AA = A;
}

struct S54 {
  S54() {
    N54::AA::foo();
  }
};
#elif defined(SECOND)
namespace N54 {
namespace B {
void foo();
}
namespace BB = B;
}

struct S54 {
  S54() {
    N54::BB::foo();
  }
};
#else
S54 s54;
// expected-error@first.h:* {{'S54' has different definitions in different modules; definition in module 'first' is here}}
// expected-note@second.h:* {{definition in module 'second' is here}}
#endif

#if defined(FIRST)
namespace N55 {
namespace A {
void foo();
}
namespace X = A;
}

struct S55 {
  S55() {
    N55::X::foo();
  }
};
#elif defined(SECOND)
namespace N55 {
namespace B {
void foo();
}
namespace X = B;
}

struct S55 {
  S55() {
    N55::X::foo();
  }
};
#else
S55 s55;
// expected-error@first.h:* {{'S55' has different definitions in different modules; definition in module 'first' is here}}
// expected-note@second.h:* {{definition in module 'second' is here}}
#endif

#if defined(FIRST)
template<int> struct foo56{};
template <template<int> class T>
struct S56 {};
struct S57 {
  S56<foo56> a;
};
#elif defined(SECOND)
template<int> struct bar56{};
template <template<int> class T>
struct S56 {};
struct S57 {
  S56<bar56> a;
};
#else
S57 s57;
// expected-error@second.h:* {{'S57::a' from module 'second' is not present in definition of 'S57' in module 'first'}}
// expected-note@first.h:* {{declaration of 'a' does not match}}
#endif

#if defined(FIRST)
template<int> struct foo58{};
template <template<int> class T>
struct S58 {};
struct S59 {
  S58<foo58> a;
};
#elif defined(SECOND)
template<int> struct foo58{};
template <template<int> class T>
struct S58 {};
struct S59 {
  S58<::foo58> a;
};
#else
S59 s59;
// expected-error@first.h:* {{'S59' has different definitions in different modules; definition in module 'first' is here}}
// expected-note@second.h:* {{definition in module 'second' is here}}
#endif


#if defined(FIRST)
struct S60a {};

typedef S60a S61;

struct S62 {
  S61 s61;
};
#elif defined(SECOND)
struct S60b {};

typedef S60b S61;

struct S62 {
  S61 s61;
};
#else
S62 s62;
// expected-error@second.h:* {{'S62::s61' from module 'second' is not present in definition of 'S62' in module 'first'}}
// expected-note@first.h:* {{declaration of 's61' does not match}}
#endif

// Don't warn on these cases
#if defined(FIRST)
void f01(int = 0);
struct S01 { friend void f01(int); };
#elif defined(SECOND)
void f01(int);
struct S01 { friend void f01(int); };
#else
S01 s01;
#endif

#if defined(FIRST)
template <template <int> class T> class Wrapper {};

template <int N> class SelfReference {
  SelfReference(Wrapper<::SelfReference> &R) {}
};

struct Xx {
  struct Yy {
  };
};

Xx::Xx::Xx::Yy yy;

namespace NNS {
template <typename> struct Foo;
template <template <class> class T = NNS::Foo>
struct NestedNamespaceSpecifier {};
}
#endif

#if defined(FIRST)
struct S02 { };
void S02Construct() {
  S02 foo;
  S02 bar = foo;
  S02 baz(bar);
}
#elif defined(SECOND)
struct S02 { };
#else
S02 s02;
#endif

#if defined(FIRST)
template <class>
struct S03 {};
#elif defined(SECOND)
template <class>
struct S03 {};
#else
S03<int> s03;
#endif

#if defined(FIRST)
template <class T>
struct S04 {
  T t;
};
#elif defined(SECOND)
template <class T>
struct S04 {
  T t;
};
#else
S03<int> s04;
#endif

#if defined(FIRST)
template <class T>
class Wrapper05;
template <class T>
struct S05 {
  Wrapper05<T> t;
};
#elif defined(SECOND)
template <class T>
class Wrapper05;
template <class T>
struct S05 {
  Wrapper05<T> t;
};
#else
template <class T>
class Wrapper05{};
S05<int> s05;
#endif

#if defined(FIRST)
enum E06 : int;
struct S06 {
  void get(E06) {}
};
#elif defined(SECOND)
enum E06 : int { A06 };
struct S06 {
  void get(E06) {}
};
#else
S06 s06;
#endif

#if defined(FIRST)
struct S07a {};
struct S07 {
  struct S07a *ptr;
};
#elif defined(SECOND)
struct S07 {
  struct S07a *ptr;
};
#else
S07 s07;
#endif

#if defined(FIRST)
struct S08a;
struct S08 {
  struct S08a *ptr;
};
#elif defined(SECOND)
struct S08 {
  struct S08a *ptr;
};
#else
S08 s08;
#endif

// TODO: Error on these cases.
#if defined(FIRST)
enum E { E1 };
#elif defined(SECOND)
enum E { E1, E2 };
#else
E e;
#endif

#if defined(FIRST)
int foo() { return 1; }
#elif defined(SECOND)
int foo() { return 2; }
#else
int number = foo();
#endif

// Keep macros contained to one file.
#ifdef FIRST
#undef FIRST
#endif
#ifdef SECOND
#undef SECOND
#endif
