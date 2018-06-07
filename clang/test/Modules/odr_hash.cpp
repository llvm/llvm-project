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

// Test that each header can compile
// RUN: %clang_cc1 -fsyntax-only -x c++ -std=c++1z %t/Inputs/first.h
// RUN: %clang_cc1 -fsyntax-only -x c++ -std=c++1z %t/Inputs/second.h

// Build module map file
// RUN: echo "module FirstModule {"     >> %t/Inputs/module.map
// RUN: echo "    header \"first.h\""   >> %t/Inputs/module.map
// RUN: echo "}"                        >> %t/Inputs/module.map
// RUN: echo "module SecondModule {"    >> %t/Inputs/module.map
// RUN: echo "    header \"second.h\""  >> %t/Inputs/module.map
// RUN: echo "}"                        >> %t/Inputs/module.map

// Run test
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/cache -x c++ -I%t/Inputs -verify %s -std=c++1z

#if !defined(FIRST) && !defined(SECOND)
#include "first.h"
#include "second.h"
#endif

// Used for testing
#if defined(FIRST)
#define ACCESS public:
#elif defined(SECOND)
#define ACCESS private:
#endif

namespace AccessSpecifiers {
#if defined(FIRST)
struct S1 {
};
#elif defined(SECOND)
struct S1 {
  private:
};
#else
S1 s1;
// expected-error@second.h:* {{'AccessSpecifiers::S1' has different definitions in different modules; first difference is definition in module 'SecondModule' found private access specifier}}
// expected-note@first.h:* {{but in 'FirstModule' found end of class}}
#endif

#if defined(FIRST)
struct S2 {
  public:
};
#elif defined(SECOND)
struct S2 {
  protected:
};
#else
S2 s2;
// expected-error@second.h:* {{'AccessSpecifiers::S2' has different definitions in different modules; first difference is definition in module 'SecondModule' found protected access specifier}}
// expected-note@first.h:* {{but in 'FirstModule' found public access specifier}}
#endif

#define DECLS \
public:       \
private:      \
protected:

#if defined(FIRST) || defined(SECOND)
struct Valid1 {
  DECLS
};
#else
Valid1 v1;
#endif

#if defined(FIRST) || defined(SECOND)
struct Invalid1 {
  DECLS
  ACCESS
};
#else
Invalid1 i1;
// expected-error@second.h:* {{'AccessSpecifiers::Invalid1' has different definitions in different modules; first difference is definition in module 'SecondModule' found private access specifier}}
// expected-note@first.h:* {{but in 'FirstModule' found public access specifier}}
#endif

#undef DECLS
} // namespace AccessSpecifiers

namespace StaticAssert {
#if defined(FIRST)
struct S1 {
  static_assert(1 == 1, "First");
};
#elif defined(SECOND)
struct S1 {
  static_assert(1 == 1, "Second");
};
#else
S1 s1;
// expected-error@second.h:* {{'StaticAssert::S1' has different definitions in different modules; first difference is definition in module 'SecondModule' found static assert with message}}
// expected-note@first.h:* {{but in 'FirstModule' found static assert with different message}}
#endif

#if defined(FIRST)
struct S2 {
  static_assert(2 == 2, "Message");
};
#elif defined(SECOND)
struct S2 {
  static_assert(2 == 2);
};
#else
S2 s2;
// expected-error@second.h:* {{'StaticAssert::S2' has different definitions in different modules; first difference is definition in module 'SecondModule' found static assert with no message}}
// expected-note@first.h:* {{but in 'FirstModule' found static assert with message}}
#endif

#if defined(FIRST)
struct S3 {
  static_assert(3 == 3, "Message");
};
#elif defined(SECOND)
struct S3 {
  static_assert(3 != 4, "Message");
};
#else
S3 s3;
// expected-error@second.h:* {{'StaticAssert::S3' has different definitions in different modules; first difference is definition in module 'SecondModule' found static assert with condition}}
// expected-note@first.h:* {{but in 'FirstModule' found static assert with different condition}}
#endif

#if defined(FIRST)
struct S4 {
  static_assert(4 == 4, "Message");
};
#elif defined(SECOND)
struct S4 {
  public:
};
#else
S4 s4;
// expected-error@second.h:* {{'StaticAssert::S4' has different definitions in different modules; first difference is definition in module 'SecondModule' found public access specifier}}
// expected-note@first.h:* {{but in 'FirstModule' found static assert}}
#endif

#define DECLS                       \
  static_assert(4 == 4, "Message"); \
  static_assert(5 == 5);

#if defined(FIRST) || defined(SECOND)
struct Valid1 {
  DECLS
};
#else
Valid1 v1;
#endif

#if defined(FIRST) || defined(SECOND)
struct Invalid1 {
  DECLS
  ACCESS
};
#else
Invalid1 i1;
// expected-error@second.h:* {{'StaticAssert::Invalid1' has different definitions in different modules; first difference is definition in module 'SecondModule' found private access specifier}}
// expected-note@first.h:* {{but in 'FirstModule' found public access specifier}}
#endif
#undef DECLS
}  // namespace StaticAssert

namespace Field {
#if defined(FIRST)
struct S1 {
  int x;
  private:
  int y;
};
#elif defined(SECOND)
struct S1 {
  int x;
  int y;
};
#else
S1 s1;
// expected-error@second.h:* {{'Field::S1' has different definitions in different modules; first difference is definition in module 'SecondModule' found field}}
// expected-note@first.h:* {{but in 'FirstModule' found private access specifier}}
#endif

#if defined(FIRST)
struct S2 {
  int x;
  int y;
};
#elif defined(SECOND)
struct S2 {
  int y;
  int x;
};
#else
S2 s2;
// expected-error@second.h:* {{'Field::S2' has different definitions in different modules; first difference is definition in module 'SecondModule' found field 'y'}}
// expected-note@first.h:* {{but in 'FirstModule' found field 'x'}}
#endif

#if defined(FIRST)
struct S3 {
  double x;
};
#elif defined(SECOND)
struct S3 {
  int x;
};
#else
S3 s3;
// expected-error@first.h:* {{'Field::S3::x' from module 'FirstModule' is not present in definition of 'Field::S3' in module 'SecondModule'}}
// expected-note@second.h:* {{declaration of 'x' does not match}}
#endif

#if defined(FIRST)
typedef int A;
struct S4 {
  A x;
};

struct S5 {
  A x;
};
#elif defined(SECOND)
typedef int B;
struct S4 {
  B x;
};

struct S5 {
  int x;
};
#else
S4 s4;
// expected-error@second.h:* {{'Field::S4' has different definitions in different modules; first difference is definition in module 'SecondModule' found field 'x' with type 'Field::B' (aka 'int')}}
// expected-note@first.h:* {{but in 'FirstModule' found field 'x' with type 'Field::A' (aka 'int')}}

S5 s5;
// expected-error@second.h:* {{'Field::S5' has different definitions in different modules; first difference is definition in module 'SecondModule' found field 'x' with type 'int'}}
// expected-note@first.h:* {{but in 'FirstModule' found field 'x' with type 'Field::A' (aka 'int')}}
#endif

#if defined(FIRST)
struct S6 {
  unsigned x;
};
#elif defined(SECOND)
struct S6 {
  unsigned x : 1;
};
#else
S6 s6;
// expected-error@second.h:* {{'Field::S6' has different definitions in different modules; first difference is definition in module 'SecondModule' found bitfield 'x'}}
// expected-note@first.h:* {{but in 'FirstModule' found non-bitfield 'x'}}
#endif

#if defined(FIRST)
struct S7 {
  unsigned x : 2;
};
#elif defined(SECOND)
struct S7 {
  unsigned x : 1;
};
#else
S7 s7;
// expected-error@second.h:* {{'Field::S7' has different definitions in different modules; first difference is definition in module 'SecondModule' found bitfield 'x' with one width expression}}
// expected-note@first.h:* {{but in 'FirstModule' found bitfield 'x' with different width expression}}
#endif

#if defined(FIRST)
struct S8 {
  unsigned x : 2;
};
#elif defined(SECOND)
struct S8 {
  unsigned x : 1 + 1;
};
#else
S8 s8;
// expected-error@second.h:* {{'Field::S8' has different definitions in different modules; first difference is definition in module 'SecondModule' found bitfield 'x' with one width expression}}
// expected-note@first.h:* {{but in 'FirstModule' found bitfield 'x' with different width expression}}
#endif

#if defined(FIRST)
struct S9 {
  mutable int x;
};
#elif defined(SECOND)
struct S9 {
  int x;
};
#else
S9 s9;
// expected-error@second.h:* {{'Field::S9' has different definitions in different modules; first difference is definition in module 'SecondModule' found non-mutable field 'x'}}
// expected-note@first.h:* {{but in 'FirstModule' found mutable field 'x'}}
#endif

#if defined(FIRST)
struct S10 {
  unsigned x = 5;
};
#elif defined(SECOND)
struct S10 {
  unsigned x;
};
#else
S10 s10;
// expected-error@second.h:* {{'Field::S10' has different definitions in different modules; first difference is definition in module 'SecondModule' found field 'x' with no initalizer}}
// expected-note@first.h:* {{but in 'FirstModule' found field 'x' with an initializer}}
#endif

#if defined(FIRST)
struct S11 {
  unsigned x = 5;
};
#elif defined(SECOND)
struct S11 {
  unsigned x = 7;
};
#else
S11 s11;
// expected-error@second.h:* {{'Field::S11' has different definitions in different modules; first difference is definition in module 'SecondModule' found field 'x' with an initializer}}
// expected-note@first.h:* {{but in 'FirstModule' found field 'x' with a different initializer}}
#endif

#if defined(FIRST)
struct S12 {
  unsigned x[5];
};
#elif defined(SECOND)
struct S12 {
  unsigned x[7];
};
#else
S12 s12;
// expected-error@first.h:* {{'Field::S12::x' from module 'FirstModule' is not present in definition of 'Field::S12' in module 'SecondModule'}}
// expected-note@second.h:* {{declaration of 'x' does not match}}
#endif

#if defined(FIRST)
struct S13 {
  unsigned x[7];
};
#elif defined(SECOND)
struct S13 {
  double x[7];
};
#else
S13 s13;
// expected-error@first.h:* {{'Field::S13::x' from module 'FirstModule' is not present in definition of 'Field::S13' in module 'SecondModule'}}
// expected-note@second.h:* {{declaration of 'x' does not match}}
#endif

#define DECLS         \
  int a;              \
  int b : 3;          \
  unsigned c : 1 + 2; \
  s d;                \
  double e = 1.0;     \
  long f[5];

#if defined(FIRST) || defined(SECOND)
typedef short s;
#endif

#if defined(FIRST) || defined(SECOND)
struct Valid1 {
  DECLS
};
#else
Valid1 v1;
#endif

#if defined(FIRST) || defined(SECOND)
struct Invalid1 {
  DECLS
  ACCESS
};
#else
Invalid1 i1;
// expected-error@second.h:* {{'Field::Invalid1' has different definitions in different modules; first difference is definition in module 'SecondModule' found private access specifier}}
// expected-note@first.h:* {{but in 'FirstModule' found public access specifier}}
#endif
#undef DECLS
}  // namespace Field

namespace Method {
#if defined(FIRST)
struct S1 {
  void A() {}
};
#elif defined(SECOND)
struct S1 {
  private:
  void A() {}
};
#else
S1 s1;
// expected-error@second.h:* {{'Method::S1' has different definitions in different modules; first difference is definition in module 'SecondModule' found private access specifier}}
// expected-note@first.h:* {{but in 'FirstModule' found method}}
#endif

#if defined(FIRST)
struct S2 {
  void A() {}
  void B() {}
};
#elif defined(SECOND)
struct S2 {
  void B() {}
  void A() {}
};
#else
S2 s2;
// expected-error@second.h:* {{'Method::S2' has different definitions in different modules; first difference is definition in module 'SecondModule' found method 'B'}}
// expected-note@first.h:* {{but in 'FirstModule' found method 'A'}}
#endif

#if defined(FIRST)
struct S3 {
  static void A() {}
  void A(int) {}
};
#elif defined(SECOND)
struct S3 {
  void A(int) {}
  static void A() {}
};
#else
S3 s3;
// expected-error@second.h:* {{'Method::S3' has different definitions in different modules; first difference is definition in module 'SecondModule' found method 'A' is not static}}
// expected-note@first.h:* {{but in 'FirstModule' found method 'A' is static}}
#endif

#if defined(FIRST)
struct S4 {
  virtual void A() {}
  void B() {}
};
#elif defined(SECOND)
struct S4 {
  void A() {}
  virtual void B() {}
};
#else
S4 s4;
// expected-error@second.h:* {{'Method::S4' has different definitions in different modules; first difference is definition in module 'SecondModule' found method 'A' is not virtual}}
// expected-note@first.h:* {{but in 'FirstModule' found method 'A' is virtual}}
#endif

#if defined(FIRST)
struct S5 {
  virtual void A() = 0;
  virtual void B() {};
};
#elif defined(SECOND)
struct S5 {
  virtual void A() {}
  virtual void B() = 0;
};
#else
S5 *s5;
// expected-error@second.h:* {{'Method::S5' has different definitions in different modules; first difference is definition in module 'SecondModule' found method 'A' is virtual}}
// expected-note@first.h:* {{but in 'FirstModule' found method 'A' is pure virtual}}
#endif

#if defined(FIRST)
struct S6 {
  inline void A() {}
};
#elif defined(SECOND)
struct S6 {
  void A() {}
};
#else
S6 s6;
// expected-error@second.h:* {{'Method::S6' has different definitions in different modules; first difference is definition in module 'SecondModule' found method 'A' is not inline}}
// expected-note@first.h:* {{but in 'FirstModule' found method 'A' is inline}}
#endif

#if defined(FIRST)
struct S7 {
  void A() volatile {}
  void A() {}
};
#elif defined(SECOND)
struct S7 {
  void A() {}
  void A() volatile {}
};
#else
S7 s7;
// expected-error@second.h:* {{'Method::S7' has different definitions in different modules; first difference is definition in module 'SecondModule' found method 'A' is not volatile}}
// expected-note@first.h:* {{but in 'FirstModule' found method 'A' is volatile}}
#endif

#if defined(FIRST)
struct S8 {
  void A() const {}
  void A() {}
};
#elif defined(SECOND)
struct S8 {
  void A() {}
  void A() const {}
};
#else
S8 s8;
// expected-error@second.h:* {{'Method::S8' has different definitions in different modules; first difference is definition in module 'SecondModule' found method 'A' is not const}}
// expected-note@first.h:* {{but in 'FirstModule' found method 'A' is const}}
#endif

#if defined(FIRST)
struct S9 {
  void A(int x) {}
  void A(int x, int y) {}
};
#elif defined(SECOND)
struct S9 {
  void A(int x, int y) {}
  void A(int x) {}
};
#else
S9 s9;
// expected-error@second.h:* {{'Method::S9' has different definitions in different modules; first difference is definition in module 'SecondModule' found method 'A' that has 2 parameters}}
// expected-note@first.h:* {{but in 'FirstModule' found method 'A' that has 1 parameter}}
#endif

#if defined(FIRST)
struct S10 {
  void A(int x) {}
  void A(float x) {}
};
#elif defined(SECOND)
struct S10 {
  void A(float x) {}
  void A(int x) {}
};
#else
S10 s10;
// expected-error@second.h:* {{'Method::S10' has different definitions in different modules; first difference is definition in module 'SecondModule' found method 'A' with 1st parameter of type 'float'}}
// expected-note@first.h:* {{but in 'FirstModule' found method 'A' with 1st parameter of type 'int'}}
#endif

#if defined(FIRST)
struct S11 {
  void A(int x);
};
#elif defined(SECOND)
struct S11 {
  void A(int y);
};
#else
S11 s11;
// expected-error@second.h:* {{'Method::S11' has different definitions in different modules; first difference is definition in module 'SecondModule' found method 'A' with 1st parameter named 'y'}}
// expected-note@first.h:* {{but in 'FirstModule' found method 'A' with 1st parameter named 'x'}}
#endif

#if defined(FIRST)
struct S12 {
  void A(int x);
};
#elif defined(SECOND)
struct S12 {
  void A(int x = 1);
};
#else
S12 s12;
// expected-error@second.h:* {{'Method::S12' has different definitions in different modules; first difference is definition in module 'SecondModule' found method 'A' with 1st parameter without a default argument}}
// expected-note@first.h:* {{but in 'FirstModule' found method 'A' with 1st parameter with a default argument}}
#endif

#if defined(FIRST)
struct S13 {
  void A(int x = 1 + 0);
};
#elif defined(SECOND)
struct S13 {
  void A(int x = 1);
};
#else
S13 s13;
// expected-error@second.h:* {{'Method::S13' has different definitions in different modules; first difference is definition in module 'SecondModule' found method 'A' with 1st parameter with a default argument}}
// expected-note@first.h:* {{but in 'FirstModule' found method 'A' with 1st parameter with a different default argument}}
#endif

#if defined(FIRST)
struct S14 {
  void A(int x[2]);
};
#elif defined(SECOND)
struct S14 {
  void A(int x[3]);
};
#else
S14 s14;
// expected-error@second.h:* {{'Method::S14' has different definitions in different modules; first difference is definition in module 'SecondModule' found method 'A' with 1st parameter of type 'int *' decayed from 'int [3]'}}
// expected-note@first.h:* {{but in 'FirstModule' found method 'A' with 1st parameter of type 'int *' decayed from 'int [2]'}}
#endif

#if defined(FIRST)
struct S15 {
  int A() { return 0; }
};
#elif defined(SECOND)
struct S15 {
  long A() { return 0; }
};
#else
S15 s15;
// expected-error@first.h:* {{'Method::S15::A' from module 'FirstModule' is not present in definition of 'Method::S15' in module 'SecondModule'}}
// expected-note@second.h:* {{declaration of 'A' does not match}}
#endif

#define DECLS            \
  void A();              \
  static void B();       \
  virtual void C();      \
  virtual void D() = 0;  \
  inline void E();       \
  void F() const;        \
  void G() volatile;     \
  void H(int x);         \
  void I(int x = 5 + 5); \
  void J(int);           \
  void K(int x[2]);      \
  int L();

#if defined(FIRST) || defined(SECOND)
struct Valid1 {
  DECLS
};
#else
Valid1* v1;
#endif

#if defined(FIRST) || defined(SECOND)
struct Invalid1 {
  DECLS
  ACCESS
};
#else
Invalid1* i1;
// expected-error@second.h:* {{'Method::Invalid1' has different definitions in different modules; first difference is definition in module 'SecondModule' found private access specifier}}
// expected-note@first.h:* {{but in 'FirstModule' found public access specifier}}
#endif
#undef DECLS
}  // namespace Method

namespace Constructor {
#if defined(FIRST)
struct S1 {
  S1() {}
  void foo() {}
};
#elif defined(SECOND)
struct S1 {
  void foo() {}
  S1() {}
};
#else
S1 s1;
// expected-error@second.h:* {{'Constructor::S1' has different definitions in different modules; first difference is definition in module 'SecondModule' found method 'foo'}}
// expected-note@first.h:* {{but in 'FirstModule' found constructor}}
#endif

#if defined(FIRST)
struct S2 {
  S2(int) {}
  S2(int, int) {}
};
#elif defined(SECOND)
struct S2 {
  S2(int, int) {}
  S2(int) {}
};
#else
S2* s2;
// expected-error@second.h:* {{'Constructor::S2' has different definitions in different modules; first difference is definition in module 'SecondModule' found constructor that has 2 parameters}}
// expected-note@first.h:* {{but in 'FirstModule' found constructor that has 1 parameter}}
#endif

#define DECLS(CLASS) \
  CLASS(int);        \
  CLASS(double);     \
  CLASS(int, int);

#if defined(FIRST) || defined(SECOND)
struct Valid1 {
  DECLS(Valid1)
};
#else
Valid1* v1;
#endif

#if defined(FIRST) || defined(SECOND)
struct Invalid1 {
  DECLS(Invalid1)
  ACCESS
};
#else
Invalid1* i1;
// expected-error@second.h:* {{'Constructor::Invalid1' has different definitions in different modules; first difference is definition in module 'SecondModule' found private access specifier}}
// expected-note@first.h:* {{but in 'FirstModule' found public access specifier}}
#endif
#undef DECLS
}  // namespace Constructor

namespace Destructor {
#if defined(FIRST)
struct S1 {
  ~S1() {}
  S1() {}
};
#elif defined(SECOND)
struct S1 {
  S1() {}
  ~S1() {}
};
#else
S1 s1;
// expected-error@second.h:* {{'Destructor::S1' has different definitions in different modules; first difference is definition in module 'SecondModule' found constructor}}
// expected-note@first.h:* {{but in 'FirstModule' found destructor}}
#endif

#if defined(FIRST)
struct S2 {
  virtual ~S2() {}
  void foo() {}
};
#elif defined(SECOND)
struct S2 {
  ~S2() {}
  virtual void foo() {}
};
#else
S2 s2;
// expected-error@second.h:* {{'Destructor::S2' has different definitions in different modules; first difference is definition in module 'SecondModule' found destructor is not virtual}}
// expected-note@first.h:* {{but in 'FirstModule' found destructor is virtual}}
#endif

#if defined(FIRST) || defined(SECOND)
struct Valid1 {
  ~Valid1();
};
#else
Valid1 v1;
#endif

#if defined(FIRST) || defined(SECOND)
struct Invalid1 {
  ~Invalid1();
  ACCESS
};
#else
Invalid1 i1;
// expected-error@second.h:* {{'Destructor::Invalid1' has different definitions in different modules; first difference is definition in module 'SecondModule' found private access specifier}}
// expected-note@first.h:* {{but in 'FirstModule' found public access specifier}}
#endif

#if defined(FIRST) || defined(SECOND)
struct Valid2 {
  virtual ~Valid2();
};
#else
Valid2 v2;
#endif

#if defined(FIRST) || defined(SECOND)
struct Invalid2 {
  virtual ~Invalid2();
  ACCESS
};
#else
Invalid2 i2;
// expected-error@second.h:* {{'Destructor::Invalid2' has different definitions in different modules; first difference is definition in module 'SecondModule' found private access specifier}}
// expected-note@first.h:* {{but in 'FirstModule' found public access specifier}}
#endif
}  // namespace Destructor

namespace TypeDef {
#if defined(FIRST)
struct S1 {
  typedef int a;
};
#elif defined(SECOND)
struct S1 {
  typedef double a;
};
#else
S1 s1;
// expected-error@first.h:* {{'TypeDef::S1::a' from module 'FirstModule' is not present in definition of 'TypeDef::S1' in module 'SecondModule'}}
// expected-note@second.h:* {{declaration of 'a' does not match}}
#endif

#if defined(FIRST)
struct S2 {
  typedef int a;
};
#elif defined(SECOND)
struct S2 {
  typedef int b;
};
#else
S2 s2;
// expected-error@first.h:* {{'TypeDef::S2::a' from module 'FirstModule' is not present in definition of 'TypeDef::S2' in module 'SecondModule'}}
// expected-note@second.h:* {{definition has no member 'a'}}
#endif

#if defined(FIRST)
typedef int T;
struct S3 {
  typedef T a;
};
#elif defined(SECOND)
typedef double T;
struct S3 {
  typedef T a;
};
#else
S3 s3;
// expected-error@first.h:* {{'TypeDef::S3::a' from module 'FirstModule' is not present in definition of 'TypeDef::S3' in module 'SecondModule'}}
// expected-note@second.h:* {{declaration of 'a' does not match}}
#endif

#if defined(FIRST)
struct S4 {
  typedef int a;
  typedef int b;
};
#elif defined(SECOND)
struct S4 {
  typedef int b;
  typedef int a;
};
#else
S4 s4;
// expected-error@second.h:* {{'TypeDef::S4' has different definitions in different modules; first difference is definition in module 'SecondModule' found typedef name 'b'}}
// expected-note@first.h:* {{but in 'FirstModule' found typedef name 'a'}}
#endif

#if defined(FIRST)
struct S5 {
  typedef int a;
  typedef int b;
  int x;
};
#elif defined(SECOND)
struct S5 {
  int x;
  typedef int b;
  typedef int a;
};
#else
S5 s5;
// expected-error@second.h:* {{'TypeDef::S5' has different definitions in different modules; first difference is definition in module 'SecondModule' found field}}
// expected-note@first.h:* {{but in 'FirstModule' found typedef}}
#endif

#if defined(FIRST)
typedef float F;
struct S6 {
  typedef int a;
  typedef F b;
};
#elif defined(SECOND)
struct S6 {
  typedef int a;
  typedef float b;
};
#else
S6 s6;
// expected-error@second.h:* {{'TypeDef::S6' has different definitions in different modules; first difference is definition in module 'SecondModule' found typedef 'b' with underlying type 'float'}}
// expected-note@first.h:* {{but in 'FirstModule' found typedef 'b' with different underlying type 'TypeDef::F' (aka 'float')}}
#endif

#define DECLS       \
  typedef int A;    \
  typedef double B; \
  typedef I C;

#if defined(FIRST) || defined(SECOND)
typedef int I;
#endif

#if defined(FIRST) || defined(SECOND)
struct Valid1 {
  DECLS
};
#else
Valid1 v1;
#endif

#if defined(FIRST) || defined(SECOND)
struct Invalid1 {
  DECLS
  ACCESS
};
#else
Invalid1 i1;
// expected-error@second.h:* {{'TypeDef::Invalid1' has different definitions in different modules; first difference is definition in module 'SecondModule' found private access specifier}}
// expected-note@first.h:* {{but in 'FirstModule' found public access specifier}}
#endif
#undef DECLS
}  // namespace TypeDef

namespace Using {
#if defined(FIRST)
struct S1 {
  using a = int;
};
#elif defined(SECOND)
struct S1 {
  using a = double;
};
#else
S1 s1;
// expected-error@first.h:* {{'Using::S1::a' from module 'FirstModule' is not present in definition of 'Using::S1' in module 'SecondModule'}}
// expected-note@second.h:* {{declaration of 'a' does not match}}
#endif

#if defined(FIRST)
struct S2 {
  using a = int;
};
#elif defined(SECOND)
struct S2 {
  using b = int;
};
#else
S2 s2;
// expected-error@first.h:* {{'Using::S2::a' from module 'FirstModule' is not present in definition of 'Using::S2' in module 'SecondModule'}}
// expected-note@second.h:* {{definition has no member 'a'}}
#endif

#if defined(FIRST)
typedef int T;
struct S3 {
  using a = T;
};
#elif defined(SECOND)
typedef double T;
struct S3 {
  using a = T;
};
#else
S3 s3;
// expected-error@first.h:* {{'Using::S3::a' from module 'FirstModule' is not present in definition of 'Using::S3' in module 'SecondModule'}}
// expected-note@second.h:* {{declaration of 'a' does not match}}
#endif

#if defined(FIRST)
struct S4 {
  using a = int;
  using b = int;
};
#elif defined(SECOND)
struct S4 {
  using b = int;
  using a = int;
};
#else
S4 s4;
// expected-error@second.h:* {{'Using::S4' has different definitions in different modules; first difference is definition in module 'SecondModule' found type alias name 'b'}}
// expected-note@first.h:* {{but in 'FirstModule' found type alias name 'a'}}
#endif

#if defined(FIRST)
struct S5 {
  using a = int;
  using b = int;
  int x;
};
#elif defined(SECOND)
struct S5 {
  int x;
  using b = int;
  using a = int;
};
#else
S5 s5;
// expected-error@second.h:* {{'Using::S5' has different definitions in different modules; first difference is definition in module 'SecondModule' found field}}
// expected-note@first.h:* {{but in 'FirstModule' found type alias}}
#endif

#if defined(FIRST)
typedef float F;
struct S6 {
  using a = int;
  using b = F;
};
#elif defined(SECOND)
struct S6 {
  using a = int;
  using b = float;
};
#else
S6 s6;
// expected-error@second.h:* {{'Using::S6' has different definitions in different modules; first difference is definition in module 'SecondModule' found type alias 'b' with underlying type 'float'}}
// expected-note@first.h:* {{but in 'FirstModule' found type alias 'b' with different underlying type 'Using::F' (aka 'float')}}
#endif

#if defined(FIRST) || defined(SECOND)
using I = int;
#endif

#define DECLS       \
  using A = int;    \
  using B = double; \
  using C = I;

#if defined(FIRST) || defined(SECOND)
struct Valid1 {
  DECLS
};
#else
Valid1 v1;
#endif

#if defined(FIRST) || defined(SECOND)
struct Invalid1 {
  DECLS
  ACCESS
};
#else
Invalid1 i1;
// expected-error@second.h:* {{'Using::Invalid1' has different definitions in different modules; first difference is definition in module 'SecondModule' found private access specifier}}
// expected-note@first.h:* {{but in 'FirstModule' found public access specifier}}
#endif
#undef DECLS
}  // namespace Using

namespace RecordType {
#if defined(FIRST)
struct B1 {};
struct S1 {
  B1 x;
};
#elif defined(SECOND)
struct A1 {};
struct S1 {
  A1 x;
};
#else
S1 s1;
// expected-error@first.h:* {{'RecordType::S1::x' from module 'FirstModule' is not present in definition of 'RecordType::S1' in module 'SecondModule'}}
// expected-note@second.h:* {{declaration of 'x' does not match}}
#endif

#define DECLS \
  Foo F;

#if defined(FIRST) || defined(SECOND)
struct Foo {};
#endif

#if defined(FIRST) || defined(SECOND)
struct Valid1 {
  DECLS
};
#else
Valid1 v1;
#endif

#if defined(FIRST) || defined(SECOND)
struct Invalid1 {
  DECLS
  ACCESS
};
#else
Invalid1 i1;
// expected-error@second.h:* {{'RecordType::Invalid1' has different definitions in different modules; first difference is definition in module 'SecondModule' found private access specifier}}
// expected-note@first.h:* {{but in 'FirstModule' found public access specifier}}
#endif
#undef DECLS
}  // namespace RecordType

namespace DependentType {
#if defined(FIRST)
template <class T>
class S1 {
  typename T::typeA x;
};
#elif defined(SECOND)
template <class T>
class S1 {
  typename T::typeB x;
};
#else
template<class T>
using U1 = S1<T>;
// expected-error@first.h:* {{'DependentType::S1::x' from module 'FirstModule' is not present in definition of 'S1<T>' in module 'SecondModule'}}
// expected-note@second.h:* {{declaration of 'x' does not match}}
#endif

#define DECLS \
  typename T::typeA x;

#if defined(FIRST) || defined(SECOND)
template <class T>
struct Valid1 {
  DECLS
};
#else
template <class T>
using V1 = Valid1<T>;
#endif

#if defined(FIRST) || defined(SECOND)
template <class T>
struct Invalid1 {
  DECLS
  ACCESS
};
#else
template <class T>
using I1 = Invalid1<T>;
// expected-error@second.h:* {{'DependentType::Invalid1' has different definitions in different modules; first difference is definition in module 'SecondModule' found private access specifier}}
// expected-note@first.h:* {{but in 'FirstModule' found public access specifier}}
#endif
#undef DECLS
}  // namespace DependentType

namespace ElaboratedType {
#if defined(FIRST)
namespace N1 { using type = double; }
struct S1 {
  N1::type x;
};
#elif defined(SECOND)
namespace N1 { using type = int; }
struct S1 {
  N1::type x;
};
#else
S1 s1;
// expected-error@first.h:* {{'ElaboratedType::S1::x' from module 'FirstModule' is not present in definition of 'ElaboratedType::S1' in module 'SecondModule'}}
// expected-note@second.h:* {{declaration of 'x' does not match}}
#endif

#define DECLS \
  NS::type x;

#if defined(FIRST) || defined(SECOND)
namespace NS { using type = float; }
#endif

#if defined(FIRST) || defined(SECOND)
struct Valid1 {
  DECLS
};
#else
Valid1 v1;
#endif

#if defined(FIRST) || defined(SECOND)
struct Invalid1 {
  DECLS
  ACCESS
};
#else
Invalid1 i1;
// expected-error@second.h:* {{'ElaboratedType::Invalid1' has different definitions in different modules; first difference is definition in module 'SecondModule' found private access specifier}}
// expected-note@first.h:* {{but in 'FirstModule' found public access specifier}}
#endif
#undef DECLS
}  // namespace ElaboratedType

namespace Enum {
#if defined(FIRST)
enum A1 {};
struct S1 {
  A1 x;
};
#elif defined(SECOND)
enum A2 {};
struct S1 {
  A2 x;
};
#else
S1 s1;
// expected-error@first.h:* {{'Enum::S1::x' from module 'FirstModule' is not present in definition of 'Enum::S1' in module 'SecondModule'}}
// expected-note@second.h:* {{declaration of 'x' does not match}}
#endif

#define DECLS \
  E e = E1;

#if defined(FIRST) || defined(SECOND)
enum E { E1, E2 };
#endif

#if defined(FIRST) || defined(SECOND)
struct Valid1 {
  DECLS
};
#else
Valid1 v1;
#endif

#if defined(FIRST) || defined(SECOND)
struct Invalid1 {
  DECLS
  ACCESS
};
#else
Invalid1 i1;
// expected-error@second.h:* {{'Enum::Invalid1' has different definitions in different modules; first difference is definition in module 'SecondModule' found private access specifier}}
// expected-note@first.h:* {{but in 'FirstModule' found public access specifier}}
#endif
#undef DECLS
}

namespace NestedNamespaceSpecifier {
#if defined(FIRST)
namespace LevelA1 {
using Type = int;
}

struct S1 {
  LevelA1::Type x;
};
# elif defined(SECOND)
namespace LevelB1 {
namespace LevelC1 {
using Type = int;
}
}

struct S1 {
  LevelB1::LevelC1::Type x;
};
#else
S1 s1;
// expected-error@second.h:* {{'NestedNamespaceSpecifier::S1' has different definitions in different modules; first difference is definition in module 'SecondModule' found field 'x' with type 'LevelB1::LevelC1::Type' (aka 'int')}}
// expected-note@first.h:* {{but in 'FirstModule' found field 'x' with type 'LevelA1::Type' (aka 'int')}}
#endif

#if defined(FIRST)
namespace LevelA2 { using Type = int; }
struct S2 {
  LevelA2::Type x;
};
# elif defined(SECOND)
struct S2 {
  int x;
};
#else
S2 s2;
// expected-error@second.h:* {{'NestedNamespaceSpecifier::S2' has different definitions in different modules; first difference is definition in module 'SecondModule' found field 'x' with type 'int'}}
// expected-note@first.h:* {{but in 'FirstModule' found field 'x' with type 'LevelA2::Type' (aka 'int')}}
#endif

namespace LevelA3 { using Type = int; }
namespace LevelB3 { using Type = int; }
#if defined(FIRST)
struct S3 {
  LevelA3::Type x;
};
# elif defined(SECOND)
struct S3 {
  LevelB3::Type x;
};
#else
S3 s3;
// expected-error@second.h:* {{'NestedNamespaceSpecifier::S3' has different definitions in different modules; first difference is definition in module 'SecondModule' found field 'x' with type 'LevelB3::Type' (aka 'int')}}
// expected-note@first.h:* {{but in 'FirstModule' found field 'x' with type 'LevelA3::Type' (aka 'int')}}
#endif

#if defined(FIRST)
struct TA4 { using Type = int; };
struct S4 {
  TA4::Type x;
};
# elif defined(SECOND)
struct TB4 { using Type = int; };
struct S4 {
  TB4::Type x;
};
#else
S4 s4;
// expected-error@second.h:* {{'NestedNamespaceSpecifier::S4' has different definitions in different modules; first difference is definition in module 'SecondModule' found field 'x' with type 'TB4::Type' (aka 'int')}}
// expected-note@first.h:* {{but in 'FirstModule' found field 'x' with type 'TA4::Type' (aka 'int')}}
#endif

#if defined(FIRST)
struct T5 { using Type = int; };
struct S5 {
  T5::Type x;
};
# elif defined(SECOND)
namespace T5 { using Type = int; };
struct S5 {
  T5::Type x;
};
#else
S5 s5;
// expected-error@second.h:* {{'NestedNamespaceSpecifier::S5' has different definitions in different modules; first difference is definition in module 'SecondModule' found field 'x' with type 'T5::Type' (aka 'int')}}
// expected-note@first.h:* {{but in 'FirstModule' found field 'x' with type 'T5::Type' (aka 'int')}}
#endif

#if defined(FIRST)
namespace N6 {using I = int;}
struct S6 {
  NestedNamespaceSpecifier::N6::I x;
};
# elif defined(SECOND)
using I = int;
struct S6 {
  ::NestedNamespaceSpecifier::I x;
};
#else
S6 s6;
// expected-error@second.h:* {{'NestedNamespaceSpecifier::S6' has different definitions in different modules; first difference is definition in module 'SecondModule' found field 'x' with type '::NestedNamespaceSpecifier::I' (aka 'int')}}
// expected-note@first.h:* {{but in 'FirstModule' found field 'x' with type 'NestedNamespaceSpecifier::N6::I' (aka 'int')}}
#endif

#if defined(FIRST)
template <class T, class U>
class S7 {
  typename T::type *x = {};
  int z = x->T::foo();
};
#elif defined(SECOND)
template <class T, class U>
class S7 {
  typename T::type *x = {};
  int z = x->U::foo();
};
#else
template <class T, class U>
using U7 = S7<T, U>;
// expected-error@second.h:* {{'NestedNamespaceSpecifier::S7' has different definitions in different modules; first difference is definition in module 'SecondModule' found field 'z' with an initializer}}
// expected-note@first.h:* {{but in 'FirstModule' found field 'z' with a different initializer}}
#endif

#if defined(FIRST)
template <class T>
class S8 {
  int x = T::template X<int>::value;
};
#elif defined(SECOND)
template <class T>
class S8 {
  int x = T::template Y<int>::value;
};
#else
template <class T>
using U8 = S8<T>;
// expected-error@second.h:* {{'NestedNamespaceSpecifier::S8' has different definitions in different modules; first difference is definition in module 'SecondModule' found field 'x' with an initializer}}
// expected-note@first.h:* {{but in 'FirstModule' found field 'x' with a different initializer}}
#endif

#if defined(FIRST)
namespace N9 { using I = int; }
namespace O9 = N9;
struct S9 {
  O9::I x;
};
#elif defined(SECOND)
namespace N9 { using I = int; }
namespace P9 = N9;
struct S9 {
  P9::I x;
};
#else
S9 s9;
// expected-error@second.h:* {{'NestedNamespaceSpecifier::S9' has different definitions in different modules; first difference is definition in module 'SecondModule' found field 'x' with type 'P9::I' (aka 'int')}}
// expected-note@first.h:* {{but in 'FirstModule' found field 'x' with type 'O9::I' (aka 'int')}}
#endif

namespace N10 {
#if defined(FIRST)
inline namespace A { struct X {}; }
struct S10 {
  A::X x;
};
#elif defined(SECOND)
inline namespace B { struct X {}; }
struct S10 {
  B::X x;
};
#else
S10 s10;
// expected-error@second.h:* {{'NestedNamespaceSpecifier::N10::S10::x' from module 'SecondModule' is not present in definition of 'NestedNamespaceSpecifier::N10::S10' in module 'FirstModule'}}
// expected-note@first.h:* {{declaration of 'x' does not match}}
#endif
}

#define DECLS       \
  NS1::Type a;      \
  NS1::NS2::Type b; \
  NS1::S c;         \
  NS3::Type d;

#if defined(FIRST) || defined(SECOND)
namespace NS1 {
  using Type = int;
  namespace NS2 {
    using Type = double;
  }
  struct S {};
}
namespace NS3 = NS1;
#endif

#if defined(FIRST) || defined(SECOND)
struct Valid1 {
  DECLS
};
#else
Valid1 v1;
#endif

#if defined(FIRST) || defined(SECOND)
struct Invalid1 {
  DECLS
  ACCESS
};
#else
Invalid1 i1;
// expected-error@second.h:* {{'NestedNamespaceSpecifier::Invalid1' has different definitions in different modules; first difference is definition in module 'SecondModule' found private access specifier}}
// expected-note@first.h:* {{but in 'FirstModule' found public access specifier}}
#endif
#undef DECLS

#define DECLS               \
  typename T::type *x = {}; \
  int y = x->T::foo();      \
  int z = U::template X<int>::value;

#if defined(FIRST) || defined(SECOND)
template <class T, class U>
struct Valid2 {
  DECLS
};
#else
template <class T, class U>
using V2 = Valid2<T, U>;
#endif

#if defined(FIRST) || defined(SECOND)
template <class T, class U>
struct Invalid2 {
  DECLS
  ACCESS
};
#else
template <class T, class U>
using I2 = Invalid2<T, U>;
// expected-error@second.h:* {{'NestedNamespaceSpecifier::Invalid2' has different definitions in different modules; first difference is definition in module 'SecondModule' found private access specifier}}
// expected-note@first.h:* {{but in 'FirstModule' found public access specifier}}
#endif
#undef DECLS
}  // namespace NestedNamespaceSpecifier

namespace TemplateSpecializationType {
#if defined(FIRST)
template <class T1> struct U1 {};
struct S1 {
  U1<int> u;
};
#elif defined(SECOND)
template <class T1, class T2> struct U1 {};
struct S1 {
  U1<int, int> u;
};
#else
S1 s1;
// expected-error@first.h:* {{'TemplateSpecializationType::S1::u' from module 'FirstModule' is not present in definition of 'TemplateSpecializationType::S1' in module 'SecondModule'}}
// expected-note@second.h:* {{declaration of 'u' does not match}}
#endif

#if defined(FIRST)
template <class T1> struct U2 {};
struct S2 {
  U2<int> u;
};
#elif defined(SECOND)
template <class T1> struct V1 {};
struct S2 {
  V1<int> u;
};
#else
S2 s2;
// expected-error@first.h:* {{'TemplateSpecializationType::S2::u' from module 'FirstModule' is not present in definition of 'TemplateSpecializationType::S2' in module 'SecondModule'}}
// expected-note@second.h:* {{declaration of 'u' does not match}}
#endif

#define DECLS                       \
  OneTemplateArg<int> x;            \
  OneTemplateArg<double> y;         \
  OneTemplateArg<char *> z;         \
  TwoTemplateArgs<int, int> a;      \
  TwoTemplateArgs<double, float> b; \
  TwoTemplateArgs<short *, char> c;

#if defined(FIRST) || defined(SECOND)
template <class T> struct OneTemplateArg {};
template <class T, class U> struct TwoTemplateArgs {};
#endif

#if defined(FIRST) || defined(SECOND)
struct Valid1 {
DECLS
};
#else
Valid1 v1;
#endif

#if defined(FIRST) || defined(SECOND)
struct Invalid1 {
DECLS
ACCESS
};
#else
Invalid1 i1;
// expected-error@second.h:* {{'TemplateSpecializationType::Invalid1' has different definitions in different modules; first difference is definition in module 'SecondModule' found private access specifier}}
// expected-note@first.h:* {{but in 'FirstModule' found public access specifier}}
#endif
#undef DECLS
}  // namespace TemplateSpecializationType

namespace TemplateArgument {
#if defined(FIRST)
template <class> struct U1{};
struct S1 {
  U1<int> x;
};
#elif defined(SECOND)
template <int> struct U1{};
struct S1 {
  U1<1> x;
};
#else
S1 s1;
// expected-error@first.h:* {{'TemplateArgument::S1::x' from module 'FirstModule' is not present in definition of 'TemplateArgument::S1' in module 'SecondModule'}}
// expected-note@second.h:* {{declaration of 'x' does not match}}
#endif

#if defined(FIRST)
template <int> struct U2{};
struct S2 {
  using T = U2<2>;
};
#elif defined(SECOND)
template <int> struct U2{};
struct S2 {
  using T = U2<(2)>;
};
#else
S2 s2;
// expected-error@second.h:* {{'TemplateArgument::S2' has different definitions in different modules; first difference is definition in module 'SecondModule' found type alias 'T' with underlying type 'U2<(2)>'}}
// expected-note@first.h:* {{but in 'FirstModule' found type alias 'T' with different underlying type 'U2<2>'}}
#endif

#if defined(FIRST)
template <int> struct U3{};
struct S3 {
  using T = U3<2>;
};
#elif defined(SECOND)
template <int> struct U3{};
struct S3 {
  using T = U3<1 + 1>;
};
#else
S3 s3;
// expected-error@second.h:* {{'TemplateArgument::S3' has different definitions in different modules; first difference is definition in module 'SecondModule' found type alias 'T' with underlying type 'U3<1 + 1>'}}
// expected-note@first.h:* {{but in 'FirstModule' found type alias 'T' with different underlying type 'U3<2>'}}
#endif

#if defined(FIRST)
template<class> struct T4a {};
template <template <class> class T> struct U4 {};
struct S4 {
  U4<T4a> x;
};
#elif defined(SECOND)
template<class> struct T4b {};
template <template <class> class T> struct U4 {};
struct S4 {
  U4<T4b> x;
};
#else
S4 s4;
// expected-error@first.h:* {{'TemplateArgument::S4::x' from module 'FirstModule' is not present in definition of 'TemplateArgument::S4' in module 'SecondModule'}}
// expected-note@second.h:* {{declaration of 'x' does not match}}
#endif

#if defined(FIRST)
template <class T> struct U5 {};
struct S5 {
  U5<int> x;
};
#elif defined(SECOND)
template <class T> struct U5 {};
struct S5 {
  U5<short> x;
};
#else
S5 s5;
// expected-error@first.h:* {{'TemplateArgument::S5::x' from module 'FirstModule' is not present in definition of 'TemplateArgument::S5' in module 'SecondModule'}}
// expected-note@second.h:* {{declaration of 'x' does not match}}
#endif

#if defined(FIRST)
template <class T> struct U6 {};
struct S6 {
  U6<int> x;
  U6<short> y;
};
#elif defined(SECOND)
template <class T> struct U6 {};
struct S6 {
  U6<short> y;
  U6<int> x;
};
#else
S6 s6;
// expected-error@second.h:* {{'TemplateArgument::S6' has different definitions in different modules; first difference is definition in module 'SecondModule' found field 'y'}}
// expected-note@first.h:* {{but in 'FirstModule' found field 'x'}}
#endif

#if defined(FIRST)
struct S7 {
  template<int> void run() {}
  template<> void run<1>() {}
};
#elif defined(SECOND)
struct S7 {
  template<int> void run() {}
  void run() {}
};
#else
S7 s7;
// expected-error@second.h:* {{'TemplateArgument::S7' has different definitions in different modules; first difference is definition in module 'SecondModule' found method 'run' with no template arguments}}
// expected-note@first.h:* {{but in 'FirstModule' found method 'run' with template arguments}}
#endif

#if defined(FIRST)
struct S8 {
  static int a, b;
  template<int&> void run() {}
  template<int&, int&> void run() {}
  template<> void run<a>() {}
};
#elif defined(SECOND)
struct S8 {
  static int a, b;
  template<int&> void run() {}
  template<int&, int&> void run() {}
  template<> void run<a, b>() {}
};
#else
S8 s8;
// expected-error@second.h:* {{'TemplateArgument::S8' has different definitions in different modules; first difference is definition in module 'SecondModule' found method 'run' with 2 template arguments}}
// expected-note@first.h:* {{but in 'FirstModule' found method 'run' with 1 template argument}}
#endif

#if defined(FIRST)
struct S9 {
  static int a, b;
  template<int&> void run() {}
  template<> void run<a>() {}
};
#elif defined(SECOND)
struct S9 {
  static int a, b;
  template<int&> void run() {}
  template<> void run<b>() {}
};
#else
S9 s9;
// expected-error@second.h:* {{'TemplateArgument::S9' has different definitions in different modules; first difference is definition in module 'SecondModule' found method 'run' with 'b' for 1st template argument}}
// expected-note@first.h:* {{but in 'FirstModule' found method 'run' with 'a' for 1st template argument}}
#endif

#if defined(FIRST)
struct S10 {
  static int a, b;
  template<int, int&...> void run() {}
  template<> void run<1, a>() {}
};
#elif defined(SECOND)
struct S10 {
  static int a, b;
  template<int, int&...> void run() {}
  template<> void run<1, b>() {}
};
#else
S10 s10;
// expected-error@second.h:* {{'TemplateArgument::S10' has different definitions in different modules; first difference is definition in module 'SecondModule' found method 'run' with 'b' for 2nd template argument}}
// expected-note@first.h:* {{but in 'FirstModule' found method 'run' with 'a' for 2nd template argument}}
#endif

#if defined(FIRST)
struct S11 {
  static int a, b;
  template<int, int&...> void run() {}
  template<> void run<1, a>() {}
};
#elif defined(SECOND)
struct S11 {
  static int a, b;
  template<int, int&...> void run() {}
  template<> void run<1, a, a>() {}
};
#else
S11 s11;
// expected-error@second.h:* {{'TemplateArgument::S11' has different definitions in different modules; first difference is definition in module 'SecondModule' found method 'run' with 3 template arguments}}
// expected-note@first.h:* {{but in 'FirstModule' found method 'run' with 2 template arguments}}
#endif

#define DECLS                   \
  OneClass<int> a;              \
  OneInt<1> b;                  \
  using c = OneClass<float>;    \
  using d = OneInt<2>;          \
  using e = OneInt<2 + 2>;      \
  OneTemplateClass<OneClass> f; \
  OneTemplateInt<OneInt> g;     \
  static int i1, i2;            \
  template <int &>              \
  void Function() {}            \
  template <int &, int &>       \
  void Function() {}            \
  template <>                   \
  void Function<i1>() {}        \
  template <>                   \
  void Function<i2>() {}        \
  template <>                   \
  void Function<i1, i2>() {}    \
  template <>                   \
  void Function<i2, i1>() {}

#if defined(FIRST) || defined(SECOND)
template <class> struct OneClass{};
template <int> struct OneInt{};
template <template <class> class> struct OneTemplateClass{};
template <template <int> class> struct OneTemplateInt{};
#endif

#if defined(FIRST) || defined(SECOND)
struct Valid1 {
DECLS
};
#else
Valid1 v1;
#endif

#if defined(FIRST) || defined(SECOND)
struct Invalid1 {
DECLS
ACCESS
};
#else
Invalid1 i1;
// expected-error@second.h:* {{'TemplateArgument::Invalid1' has different definitions in different modules; first difference is definition in module 'SecondModule' found private access specifier}}
// expected-note@first.h:* {{but in 'FirstModule' found public access specifier}}
#endif
#undef DECLS
}  // namespace TemplateArgument

namespace TemplateTypeParmType {
#if defined(FIRST)
template <class T1, class T2>
struct S1 {
  T1 x;
};
#elif defined(SECOND)
template <class T1, class T2>
struct S1 {
  T2 x;
};
#else
using TemplateTypeParmType::S1;
// expected-error@first.h:* {{'TemplateTypeParmType::S1::x' from module 'FirstModule' is not present in definition of 'S1<T1, T2>' in module 'SecondModule'}}
// expected-note@second.h:* {{declaration of 'x' does not match}}
#endif

#if defined(FIRST)
template <int ...Ts>
struct U2 {};
template <int T, int U>
class S2 {
  typedef U2<U, T> type;
  type x;
};
#elif defined(SECOND)
template <int ...Ts>
struct U2 {};
template <int T, int U>
class S2 {
  typedef U2<T, U> type;
  type x;
};
#else
using TemplateTypeParmType::S2;
// expected-error@first.h:* {{'TemplateTypeParmType::S2::x' from module 'FirstModule' is not present in definition of 'S2<T, U>' in module 'SecondModule'}}
// expected-note@second.h:* {{declaration of 'x' does not match}}
// expected-error@first.h:* {{'TemplateTypeParmType::S2::type' from module 'FirstModule' is not present in definition of 'S2<T, U>' in module 'SecondModule'}}
// expected-note@second.h:* {{declaration of 'type' does not match}}
#endif

#define DECLS            \
  T t;                   \
  U u;                   \
  ParameterPack<T> a;    \
  ParameterPack<T, U> b; \
  ParameterPack<U> c;    \
  ParameterPack<U, T> d;

#if defined(FIRST) || defined(SECOND)
template <class ...Ts> struct ParameterPack {};
#endif

#if defined(FIRST) || defined(SECOND)
template <class T, class U>
struct Valid1 {
  DECLS
};
#else
using TemplateTypeParmType::Valid1;
#endif

#if defined(FIRST) || defined(SECOND)
template <class T, class U>
struct Invalid1 {
  DECLS
  ACCESS
};
#else
using TemplateTypeParmType::Invalid1;
// expected-error@second.h:* {{'TemplateTypeParmType::Invalid1' has different definitions in different modules; first difference is definition in module 'SecondModule' found private access specifier}}
// expected-note@first.h:* {{but in 'FirstModule' found public access specifier}}
#endif
#undef DECLS
}  // namespace TemplateTypeParmType

namespace VarDecl {
#if defined(FIRST)
struct S1 {
  static int x;
  static int y;
};
#elif defined(SECOND)
struct S1 {
  static int y;
  static int x;
};
#else
S1 s1;
// expected-error@second.h:* {{'VarDecl::S1' has different definitions in different modules; first difference is definition in module 'SecondModule' found data member with name 'y'}}
// expected-note@first.h:* {{but in 'FirstModule' found data member with name 'x'}}
#endif

#if defined(FIRST)
struct S2 {
  static int x;
};
#elif defined(SECOND)
using I = int;
struct S2 {
  static I x;
};
#else
S2 s2;
// expected-error@second.h:* {{'VarDecl::S2' has different definitions in different modules; first difference is definition in module 'SecondModule' found data member 'x' with type 'VarDecl::I' (aka 'int')}}
// expected-note@first.h:* {{but in 'FirstModule' found data member 'x' with different type 'int'}}
#endif

#if defined(FIRST)
struct S3 {
  static const int x = 1;
};
#elif defined(SECOND)
struct S3 {
  static const int x;
};
#else
S3 s3;
// expected-error@second.h:* {{'VarDecl::S3' has different definitions in different modules; first difference is definition in module 'SecondModule' found data member 'x' with an initializer}}
// expected-note@first.h:* {{but in 'FirstModule' found data member 'x' without an initializer}}
#endif

#if defined(FIRST)
struct S4 {
  static const int x = 1;
};
#elif defined(SECOND)
struct S4 {
  static const int x = 2;
};
#else
S4 s4;
// expected-error@second.h:* {{'VarDecl::S4' has different definitions in different modules; first difference is definition in module 'SecondModule' found data member 'x' with an initializer}}
// expected-note@first.h:* {{but in 'FirstModule' found data member 'x' with a different initializer}}
#endif

#if defined(FIRST)
struct S5 {
  static const int x = 1;
};
#elif defined(SECOND)
struct S5 {
  static constexpr int x = 1;
};
#else
S5 s5;
// expected-error@second.h:* {{'VarDecl::S5' has different definitions in different modules; first difference is definition in module 'SecondModule' found data member 'x' is not constexpr}}
// expected-note@first.h:* {{but in 'FirstModule' found data member 'x' is constexpr}}
#endif

#if defined(FIRST)
struct S6 {
  static const int x = 1;
};
#elif defined(SECOND)
struct S6 {
  static const int y = 1;
};
#else
S6 s6;
// expected-error@first.h:* {{'VarDecl::S6::x' from module 'FirstModule' is not present in definition of 'VarDecl::S6' in module 'SecondModule'}}
// expected-note@second.h:* {{definition has no member 'x'}}
#endif

#if defined(FIRST)
struct S7 {
  static const int x = 1;
};
#elif defined(SECOND)
struct S7 {
  static const unsigned x = 1;
};
#else
S7 s7;
// expected-error@first.h:* {{'VarDecl::S7::x' from module 'FirstModule' is not present in definition of 'VarDecl::S7' in module 'SecondModule'}}
// expected-note@second.h:* {{declaration of 'x' does not match}}
#endif

#if defined(FIRST)
struct S8 {
public:
  static const int x = 1;
};
#elif defined(SECOND)
struct S8 {
  static const int x = 1;
public:
};
#else
S8 s8;
// expected-error@second.h:* {{'VarDecl::S8' has different definitions in different modules; first difference is definition in module 'SecondModule' found data member}}
// expected-note@first.h:* {{but in 'FirstModule' found public access specifier}}
#endif

#if defined(FIRST)
struct S9 {
  static const int x = 1;
};
#elif defined(SECOND)
struct S9 {
  static int x;
};
#else
S9 s9;
// expected-error@first.h:* {{'VarDecl::S9::x' from module 'FirstModule' is not present in definition of 'VarDecl::S9' in module 'SecondModule'}}
// expected-note@second.h:* {{declaration of 'x' does not match}}
#endif

#define DECLS             \
  static int a;           \
  static I b;             \
  static const int c = 1; \
  static constexpr int d = 5;

#if defined(FIRST) || defined(SECOND)
using I = int;
#endif

#if defined(FIRST) || defined(SECOND)
struct Valid1 {
  DECLS
};
#else
Valid1 v1;
#endif

#if defined(FIRST) || defined(SECOND)
struct Invalid1 {
  DECLS
  ACCESS
};
#else
Invalid1 i1;
// expected-error@second.h:* {{'VarDecl::Invalid1' has different definitions in different modules; first difference is definition in module 'SecondModule' found private access specifier}}
// expected-note@first.h:* {{but in 'FirstModule' found public access specifier}}
#endif
#undef DECLS
}  // namespace VarDecl

namespace Friend {
#if defined(FIRST)
struct T1 {};
struct S1 {
  friend class T1;
};
#elif defined(SECOND)
struct T1 {};
struct S1 {
  friend T1;
};
#else
S1 s1;
// expected-error@second.h:* {{'Friend::S1' has different definitions in different modules; first difference is definition in module 'SecondModule' found friend 'Friend::T1'}}
// expected-note@first.h:* {{but in 'FirstModule' found friend 'class T1'}}
#endif

#if defined(FIRST)
struct T2 {};
struct S2 {
  friend class T2;
};
#elif defined(SECOND)
struct T2 {};
struct S2 {
  friend struct T2;
};
#else
S2 s2;
// expected-error@second.h:* {{'Friend::S2' has different definitions in different modules; first difference is definition in module 'SecondModule' found friend 'struct T2'}}
// expected-note@first.h:* {{but in 'FirstModule' found friend 'class T2'}}
#endif

#if defined(FIRST)
struct T3 {};
struct S3 {
  friend const T3;
};
#elif defined(SECOND)
struct T3 {};
struct S3 {
  friend T3;
};
#else
S3 s3;
// expected-error@second.h:* {{'Friend::S3' has different definitions in different modules; first difference is definition in module 'SecondModule' found friend 'Friend::T3'}}
// expected-note@first.h:* {{but in 'FirstModule' found friend 'const Friend::T3'}}
#endif

#if defined(FIRST)
struct T4 {};
struct S4 {
  friend T4;
};
#elif defined(SECOND)
struct S4 {
  friend void T4();
};
#else
S4 s4;
// expected-error@second.h:* {{'Friend::S4' has different definitions in different modules; first difference is definition in module 'SecondModule' found friend function}}
// expected-note@first.h:* {{but in 'FirstModule' found friend class}}
#endif

#if defined(FIRST)
struct S5 {
  friend void T5a();
};
#elif defined(SECOND)
struct S5 {
  friend void T5b();
};
#else
S5 s5;
// expected-error@second.h:* {{'Friend::S5' has different definitions in different modules; first difference is definition in module 'SecondModule' found friend function 'T5b'}}
// expected-note@first.h:* {{but in 'FirstModule' found friend function 'T5a'}}
#endif

#define DECLS            \
  friend class FriendA;  \
  friend struct FriendB; \
  friend FriendC;        \
  friend const FriendD;  \
  friend void Function();

#if defined(FIRST) || defined(SECOND)
class FriendA {};
class FriendB {};
class FriendC {};
class FriendD {};
#endif

#if defined(FIRST) || defined(SECOND)
struct Valid1 {
  DECLS
};
#else
Valid1 v1;
#endif

#if defined(FIRST) || defined(SECOND)
struct Invalid1 {
  DECLS
  ACCESS
};
#else
Invalid1 i1;
// expected-error@second.h:* {{'Friend::Invalid1' has different definitions in different modules; first difference is definition in module 'SecondModule' found private access specifier}}
// expected-note@first.h:* {{but in 'FirstModule' found public access specifier}}
#endif
#undef DECLS
}  // namespace Friend

namespace TemplateParameters {
#if defined(FIRST)
template <class A>
struct S1 {};
#elif defined(SECOND)
template <class B>
struct S1 {};
#else
using TemplateParameters::S1;
// expected-error@second.h:* {{'TemplateParameters::S1' has different definitions in different modules; first difference is definition in module 'SecondModule' found template parameter 'B'}}
// expected-note@first.h:* {{but in 'FirstModule' found template parameter 'A'}}
#endif

#if defined(FIRST)
template <class A = double>
struct S2 {};
#elif defined(SECOND)
template <class A = int>
struct S2 {};
#else
using TemplateParameters::S2;
// expected-error@second.h:* {{'TemplateParameters::S2' has different definitions in different modules; first difference is definition in module 'SecondModule' found template parameter with default argument}}
// expected-note@first.h:* {{but in 'FirstModule' found template parameter with different default argument}}
#endif

#if defined(FIRST)
template <class A = int>
struct S3 {};
#elif defined(SECOND)
template <class A>
struct S3 {};
#else
using TemplateParameters::S3;
// expected-error@second.h:* {{'TemplateParameters::S3' has different definitions in different modules; first difference is definition in module 'SecondModule' found template parameter with no default argument}}
// expected-note@first.h:* {{but in 'FirstModule' found template parameter with default argument}}
#endif

#if defined(FIRST)
template <int A>
struct S4 {};
#elif defined(SECOND)
template <int A = 2>
struct S4 {};
#else
using TemplateParameters::S4;
// expected-error@second.h:* {{'TemplateParameters::S4' has different definitions in different modules; first difference is definition in module 'SecondModule' found template parameter with default argument}}
// expected-note@first.h:* {{but in 'FirstModule' found template parameter with no default argument}}
#endif

#if defined(FIRST)
template <int> class S5_first {};
template <template<int> class A = S5_first>
struct S5 {};
#elif defined(SECOND)
template <int> class S5_second {};
template <template<int> class A = S5_second>
struct S5 {};
#else
using TemplateParameters::S5;
// expected-error@second.h:* {{'TemplateParameters::S5' has different definitions in different modules; first difference is definition in module 'SecondModule' found template parameter with default argument}}
// expected-note@first.h:* {{but in 'FirstModule' found template parameter with different default argument}}
#endif

#if defined(FIRST)
template <class A>
struct S6 {};
#elif defined(SECOND)
template <class>
struct S6 {};
#else
using TemplateParameters::S6;
// expected-error@second.h:* {{'TemplateParameters::S6' has different definitions in different modules; first difference is definition in module 'SecondModule' found unnamed template parameter}}
// expected-note@first.h:* {{but in 'FirstModule' found template parameter 'A'}}
#endif

#define DECLS

#if defined(FIRST) || defined(SECOND)
template <class> class DefaultArg;
#endif

#if defined(FIRST) || defined(SECOND)
template <int, class, template <class> class,
          int A, class B, template <int> class C,
          int D = 1, class E = int, template <class F> class = DefaultArg>
struct Valid1 {
  DECLS
};
#else
using TemplateParameters::Valid1;
#endif

#if defined(FIRST) || defined(SECOND)
template <int, class, template <class> class,
          int A, class B, template <int> class C,
          int D = 1, class E = int, template <class F> class = DefaultArg>
struct Invalid1 {
  DECLS
  ACCESS
};
#else
using TemplateParameters::Invalid1;
// expected-error@second.h:* {{'TemplateParameters::Invalid1' has different definitions in different modules; first difference is definition in module 'SecondModule' found private access specifier}}
// expected-note@first.h:* {{but in 'FirstModule' found public access specifier}}
#endif
#undef DECLS
}  // namespace TemplateParameters

namespace BaseClass {
#if defined(FIRST)
struct B1 {};
struct S1 : B1 {};
#elif defined(SECOND)
struct S1 {};
#else
S1 s1;
// expected-error@second.h:* {{'BaseClass::S1' has different definitions in different modules; first difference is definition in module 'SecondModule' found 0 base classes}}
// expected-note@first.h:* {{but in 'FirstModule' found 1 base class}}
#endif

#if defined(FIRST)
struct S2 {};
#elif defined(SECOND)
struct B2 {};
struct S2 : virtual B2 {};
#else
S2 s2;
// expected-error@second.h:* {{'BaseClass::S2' has different definitions in different modules; first difference is definition in module 'SecondModule' found 1 base class}}
// expected-note@first.h:* {{but in 'FirstModule' found 0 base classes}}
#endif

#if defined(FIRST)
struct B3a {};
struct S3 : B3a {};
#elif defined(SECOND)
struct B3b {};
struct S3 : virtual B3b {};
#else
S3 s3;
// expected-error@second.h:* {{'BaseClass::S3' has different definitions in different modules; first difference is definition in module 'SecondModule' found 1 virtual base class}}
// expected-note@first.h:* {{but in 'FirstModule' found 0 virtual base classes}}
#endif

#if defined(FIRST)
struct B4a {};
struct S4 : B4a {};
#elif defined(SECOND)
struct B4b {};
struct S4 : B4b {};
#else
S4 s4;
// expected-error@second.h:* {{'BaseClass::S4' has different definitions in different modules; first difference is definition in module 'SecondModule' found 1st base class with type 'BaseClass::B4b'}}
// expected-note@first.h:* {{but in 'FirstModule' found 1st base class with different type 'BaseClass::B4a'}}
#endif

#if defined(FIRST)
struct B5a {};
struct S5 : virtual B5a {};
#elif defined(SECOND)
struct B5a {};
struct S5 : B5a {};
#else
S5 s5;
// expected-error@second.h:* {{'BaseClass::S5' has different definitions in different modules; first difference is definition in module 'SecondModule' found 0 virtual base classes}}
// expected-note@first.h:* {{but in 'FirstModule' found 1 virtual base class}}
#endif

#if defined(FIRST)
struct B6a {};
struct S6 : B6a {};
#elif defined(SECOND)
struct B6a {};
struct S6 : virtual B6a {};
#else
S6 s6;
// expected-error@second.h:* {{'BaseClass::S6' has different definitions in different modules; first difference is definition in module 'SecondModule' found 1 virtual base class}}
// expected-note@first.h:* {{but in 'FirstModule' found 0 virtual base classes}}
#endif

#if defined(FIRST)
struct B7a {};
struct S7 : protected B7a {};
#elif defined(SECOND)
struct B7a {};
struct S7 : B7a {};
#else
S7 s7;
// expected-error@second.h:* {{'BaseClass::S7' has different definitions in different modules; first difference is definition in module 'SecondModule' found 1st base class 'BaseClass::B7a' with no access specifier}}
// expected-note@first.h:* {{but in 'FirstModule' found 1st base class 'BaseClass::B7a' with protected access specifier}}
#endif

#if defined(FIRST)
struct B8a {};
struct S8 : public B8a {};
#elif defined(SECOND)
struct B8a {};
struct S8 : private B8a {};
#else
S8 s8;
// expected-error@second.h:* {{'BaseClass::S8' has different definitions in different modules; first difference is definition in module 'SecondModule' found 1st base class 'BaseClass::B8a' with private access specifier}}
// expected-note@first.h:* {{but in 'FirstModule' found 1st base class 'BaseClass::B8a' with public access specifier}}
#endif

#if defined(FIRST)
struct B9a {};
struct S9 : private B9a {};
#elif defined(SECOND)
struct B9a {};
struct S9 : public B9a {};
#else
S9 s9;
// expected-error@second.h:* {{'BaseClass::S9' has different definitions in different modules; first difference is definition in module 'SecondModule' found 1st base class 'BaseClass::B9a' with public access specifier}}
// expected-note@first.h:* {{but in 'FirstModule' found 1st base class 'BaseClass::B9a' with private access specifier}}
#endif

#if defined(FIRST)
struct B10a {};
struct S10 : B10a {};
#elif defined(SECOND)
struct B10a {};
struct S10 : protected B10a {};
#else
S10 s10;
// expected-error@second.h:* {{'BaseClass::S10' has different definitions in different modules; first difference is definition in module 'SecondModule' found 1st base class 'BaseClass::B10a' with protected access specifier}}
// expected-note@first.h:* {{but in 'FirstModule' found 1st base class 'BaseClass::B10a' with no access specifier}}
#endif

#define DECLS

#if defined(FIRST) || defined(SECOND)
struct Base1 {};
struct Base2 {};
struct Base3 {};
struct Base4 {};
struct Base5 {};
#endif

#if defined(FIRST) || defined(SECOND)
struct Valid1 :
  Base1, virtual Base2, protected Base3, public Base4, private Base5 {

  DECLS
};
#else
Valid1 v1;
#endif

#if defined(FIRST) || defined(SECOND)
struct Invalid1 :
  Base1, virtual Base2, protected Base3, public Base4, private Base5 {

  DECLS
  ACCESS
};
#else
Invalid1 i1;
// expected-error@second.h:* {{'BaseClass::Invalid1' has different definitions in different modules; first difference is definition in module 'SecondModule' found private access specifier}}
// expected-note@first.h:* {{but in 'FirstModule' found public access specifier}}
#endif
#undef DECLS
}  // namespace BaseClass

namespace PointersAndReferences {
#if defined(FIRST) || defined(SECOND)
template<typename> struct Wrapper{};
#endif

#if defined(FIRST)
struct S1 {
  Wrapper<int*> x;
};
#elif defined(SECOND)
struct S1 {
  Wrapper<float*> x;
};
#else
S1 s1;
// expected-error@first.h:* {{PointersAndReferences::S1::x' from module 'FirstModule' is not present in definition of 'PointersAndReferences::S1' in module 'SecondModule'}}
// expected-note@second.h:* {{declaration of 'x' does not match}}
#endif

#if defined(FIRST)
struct S2 {
  Wrapper<int &&> x;
};
#elif defined(SECOND)
struct S2 {
  Wrapper<float &&> x;
};
#else
S2 s2;
// expected-error@first.h:* {{PointersAndReferences::S2::x' from module 'FirstModule' is not present in definition of 'PointersAndReferences::S2' in module 'SecondModule'}}
// expected-note@second.h:* {{declaration of 'x' does not match}}
#endif

#if defined(FIRST)
struct S3 {
  Wrapper<int *> x;
};
#elif defined(SECOND)
struct S3 {
  Wrapper<float *> x;
};
#else
S3 s3;
// expected-error@first.h:* {{PointersAndReferences::S3::x' from module 'FirstModule' is not present in definition of 'PointersAndReferences::S3' in module 'SecondModule'}}
// expected-note@second.h:* {{declaration of 'x' does not match}}
#endif

#if defined(FIRST)
struct S4 {
  Wrapper<int &> x;
};
#elif defined(SECOND)
struct S4 {
  Wrapper<float &> x;
};
#else
S4 s4;
// expected-error@first.h:* {{PointersAndReferences::S4::x' from module 'FirstModule' is not present in definition of 'PointersAndReferences::S4' in module 'SecondModule'}}
// expected-note@second.h:* {{declaration of 'x' does not match}}
#endif

#if defined(FIRST)
struct S5 {
  Wrapper<S5 *> x;
};
#elif defined(SECOND)
struct S5 {
  Wrapper<const S5 *> x;
};
#else
S5 s5;
// expected-error@second.h:* {{'PointersAndReferences::S5::x' from module 'SecondModule' is not present in definition of 'PointersAndReferences::S5' in module 'FirstModule'}}
// expected-note@first.h:* {{declaration of 'x' does not match}}
#endif

#if defined(FIRST)
struct S6 {
  Wrapper<int &> x;
};
#elif defined(SECOND)
struct S6 {
  Wrapper<const int &> x;
};
#else
S6 s6;
// expected-error@first.h:* {{PointersAndReferences::S6::x' from module 'FirstModule' is not present in definition of 'PointersAndReferences::S6' in module 'SecondModule'}}
// expected-note@second.h:* {{declaration of 'x' does not match}}
#endif

#define DECLS                \
  Wrapper<int *> x1;         \
  Wrapper<float *> x2;       \
  Wrapper<const float *> x3; \
  Wrapper<int &> x4;         \
  Wrapper<int &&> x5;        \
  Wrapper<const int &> x6;   \
  Wrapper<S1 *> x7;          \
  Wrapper<S1 &> x8;          \
  Wrapper<S1 &&> x9;

#if defined(FIRST) || defined(SECOND)
struct Valid1 {
  DECLS
};
#else
Valid1 v1;
#endif

#if defined(FIRST) || defined(SECOND)
struct Invalid1 {
  DECLS
  ACCESS
};
#else
Invalid1 i1;
// expected-error@second.h:* {{'PointersAndReferences::Invalid1' has different definitions in different modules; first difference is definition in module 'SecondModule' found private access specifier}}
// expected-note@first.h:* {{but in 'FirstModule' found public access specifier}}
#endif
#undef DECLS
}  // namespace PointersAndReferences

namespace FunctionTemplate {
#if defined(FIRST)
struct S1 {
  template <int, int> void foo();
};
#elif defined(SECOND)
struct S1 {
  template <int> void foo();
};
#else
S1 s1;
// expected-error@first.h:* {{'FunctionTemplate::S1::foo' from module 'FirstModule' is not present in definition of 'FunctionTemplate::S1' in module 'SecondModule'}}
// expected-note@second.h:* {{declaration of 'foo' does not match}}
#endif

#if defined(FIRST)
struct S2 {
  template <char> void foo();
};
#elif defined(SECOND)
struct S2 {
  template <int> void foo();
};
#else
S2 s2;
// expected-error@first.h:* {{'FunctionTemplate::S2::foo' from module 'FirstModule' is not present in definition of 'FunctionTemplate::S2' in module 'SecondModule'}}
// expected-note@second.h:* {{declaration of 'foo' does not match}}
#endif

#if defined(FIRST)
struct S3 {
  template <int x> void foo();
};
#elif defined(SECOND)
struct S3 {
  template <int y> void foo();
};
#else
S3 s3;
// expected-error@second.h:* {{'FunctionTemplate::S3' has different definitions in different modules; first difference is definition in module 'SecondModule' found function template 'foo' with 1st template parameter named 'y'}}
// expected-note@first.h:* {{but in 'FirstModule' found function template 'foo' with 1st template parameter named 'x'}}
#endif

#if defined(FIRST)
struct S4 {
  template <int x> void foo();
};
#elif defined(SECOND)
struct S4 {
  template <int x> void bar();
};
#else
S4 s4;
// expected-error@first.h:* {{'FunctionTemplate::S4::foo' from module 'FirstModule' is not present in definition of 'FunctionTemplate::S4' in module 'SecondModule'}}
// expected-note@second.h:* {{definition has no member 'foo'}}
#endif

#if defined(FIRST)
struct S5 {
  template <int x> void foo();
};
#elif defined(SECOND)
struct S5 {
 public:
  template <int x> void foo();
};
#else
S5 s5;
// expected-error@second.h:* {{'FunctionTemplate::S5' has different definitions in different modules; first difference is definition in module 'SecondModule' found public access specifier}}
// expected-note@first.h:* {{but in 'FirstModule' found function template}}
#endif

#if defined(FIRST)
struct S6 {
  template <typename x = int> void foo();
};
#elif defined(SECOND)
struct S6 {
  template <typename x> void foo();
};
#else
S6 s6;
// expected-error@second.h:* {{'FunctionTemplate::S6' has different definitions in different modules; first difference is definition in module 'SecondModule' found function template 'foo' with 1st template parameter with no default argument}}
// expected-note@first.h:* {{but in 'FirstModule' found function template 'foo' with 1st template parameter with default argument}}
#endif

#if defined(FIRST)
struct S7 {
  template <typename x = void> void foo();
};
#elif defined(SECOND)
struct S7 {
  template <typename x = int> void foo();
};
#else
S7 s7;
// expected-error@second.h:* {{'FunctionTemplate::S7' has different definitions in different modules; first difference is definition in module 'SecondModule' found function template 'foo' with 1st template parameter with default argument 'int'}}
// expected-note@first.h:* {{but in 'FirstModule' found function template 'foo' with 1st template parameter with default argument 'void'}}
#endif

#if defined(FIRST)
template <int>
struct U8 {};
struct S8 {
  template <template<int> class x = U8> void foo();
};
#elif defined(SECOND)
template <int>
struct T8 {};
struct S8{
  template <template<int> class x = T8> void foo();
};
#else
S8 s8;
// expected-error@second.h:* {{'FunctionTemplate::S8' has different definitions in different modules; first difference is definition in module 'SecondModule' found function template 'foo' with 1st template parameter with default argument 'T8'}}
// expected-note@first.h:* {{but in 'FirstModule' found function template 'foo' with 1st template parameter with default argument 'U8'}}
#endif

#if defined(FIRST)
template <int>
struct U9 {};
struct S9 { S9();
  template <template<int> class x = U9> void foo();
};
#elif defined(SECOND)
struct S9 { S9();
  template <template<int> class x> void foo();
};
#else
S9 s9;
// expected-error@second.h:* {{'FunctionTemplate::S9' has different definitions in different modules; first difference is definition in module 'SecondModule' found function template 'foo' with 1st template parameter with no default argument}}
// expected-note@first.h:* {{but in 'FirstModule' found function template 'foo' with 1st template parameter with default argument}}
#endif

#if defined(FIRST)
struct S10 {
  template <template<int> class x> void foo();
  template <template<typename> class x> void foo();
};
#elif defined(SECOND)
struct S10 {
  template <template<typename> class x> void foo();
  template <template<int> class x> void foo();
};
#else
S10 s10;
// expected-error@second.h:* {{'FunctionTemplate::S10' has different definitions in different modules; first difference is definition in module 'SecondModule' found function template 'foo' with 1st template parameter with one type}}
// expected-note@first.h:* {{but in 'FirstModule' found function template 'foo' with 1st template parameter with different type}}
#endif

#if defined(FIRST)
struct S11 {
  template <template<int> class x> void foo();
};
#elif defined(SECOND)
struct S11 {
  template <template<int> class> void foo();
};
#else
S11 s11;
// expected-error@second.h:* {{'FunctionTemplate::S11' has different definitions in different modules; first difference is definition in module 'SecondModule' found function template 'foo' with 1st template parameter with no name}}
// expected-note@first.h:* {{but in 'FirstModule' found function template 'foo' with 1st template parameter named 'x'}}
#endif

#if defined(FIRST)
struct S12 {
  template <class> void foo();
  template <class, class> void foo();
};
#elif defined(SECOND)
struct S12 {
  template <class, class> void foo();
  template <class> void foo();
};
#else
S12 s12;
// expected-error@second.h:* {{'FunctionTemplate::S12' has different definitions in different modules; first difference is definition in module 'SecondModule' found function template 'foo' with 2 template parameters}}
// expected-note@first.h:* {{but in 'FirstModule' found function template 'foo' with 1 template parameter}}
#endif

#if defined(FIRST)
struct S13 {
  template <class = int> void foo();
};
#elif defined(SECOND)
struct S13 {
  template <class = void> void foo();
};
#else
S13 s13;
// expected-error@second.h:* {{'FunctionTemplate::S13' has different definitions in different modules; first difference is definition in module 'SecondModule' found function template 'foo' with 1st template parameter with default argument 'void'}}
// expected-note@first.h:* {{but in 'FirstModule' found function template 'foo' with 1st template parameter with default argument 'int'}}
#endif

#if defined(FIRST)
struct S14 {
  template <class = void> void foo();
};
#elif defined(SECOND)
struct S14 {
  template <class> void foo();
};
#else
S14 s14;
// expected-error@second.h:* {{'FunctionTemplate::S14' has different definitions in different modules; first difference is definition in module 'SecondModule' found function template 'foo' with 1st template parameter with no default argument}}
// expected-note@first.h:* {{but in 'FirstModule' found function template 'foo' with 1st template parameter with default argument}}
#endif

#if defined(FIRST)
struct S15 {
  template <class> void foo();
};
#elif defined(SECOND)
struct S15 {
  template <class = void> void foo();
};
#else
S15 s15;
// expected-error@second.h:* {{'FunctionTemplate::S15' has different definitions in different modules; first difference is definition in module 'SecondModule' found function template 'foo' with 1st template parameter with default argument}}
// expected-note@first.h:* {{but in 'FirstModule' found function template 'foo' with 1st template parameter with no default argument}}
#endif

#if defined(FIRST)
struct S16 {
  template <short> void foo();
};
#elif defined(SECOND)
struct S16 {
  template <short = 1> void foo();
};
#else
S16 s16;
// expected-error@second.h:* {{'FunctionTemplate::S16' has different definitions in different modules; first difference is definition in module 'SecondModule' found function template 'foo' with 1st template parameter with default argument}}
// expected-note@first.h:* {{but in 'FirstModule' found function template 'foo' with 1st template parameter with no default argument}}
#endif

#if defined(FIRST)
struct S17 {
  template <short = 2> void foo();
};
#elif defined(SECOND)
struct S17 {
  template <short = 1 + 1> void foo();
};
#else
S17 s17;
// expected-error@second.h:* {{'FunctionTemplate::S17' has different definitions in different modules; first difference is definition in module 'SecondModule' found function template 'foo' with 1st template parameter with default argument 1 + 1}}
// expected-note@first.h:* {{but in 'FirstModule' found function template 'foo' with 1st template parameter with default argument 2}}
#endif

#if defined(FIRST)
struct S18 {
  template <short> void foo();
  template <int> void foo();
};
#elif defined(SECOND)
struct S18 {
  template <int> void foo();
  template <short> void foo();
};
#else
S18 s18;
// expected-error@second.h:* {{'FunctionTemplate::S18' has different definitions in different modules; first difference is definition in module 'SecondModule' found function template 'foo' with 1st template parameter with one type}}
// expected-note@first.h:* {{but in 'FirstModule' found function template 'foo' with 1st template parameter with different type}}
#endif

#if defined(FIRST)
struct S19 {
  template <short> void foo();
  template <short...> void foo();
};
#elif defined(SECOND)
struct S19 {
  template <short...> void foo();
  template <short> void foo();
};
#else
S19 s19;
// expected-error@second.h:* {{'FunctionTemplate::S19' has different definitions in different modules; first difference is definition in module 'SecondModule' found function template 'foo' with 1st template parameter being a template parameter pack}}
// expected-note@first.h:* {{but in 'FirstModule' found function template 'foo' with 1st template parameter not being a template parameter pack}}
#endif

#if defined(FIRST)
struct S20 {
  template <class> void foo();
  template <class...> void foo();
};
#elif defined(SECOND)
struct S20 {
  template <class...> void foo();
  template <class> void foo();
};
#else
S20 s20;
// expected-error@second.h:* {{'FunctionTemplate::S20' has different definitions in different modules; first difference is definition in module 'SecondModule' found function template 'foo' with 1st template parameter being a template parameter pack}}
// expected-note@first.h:* {{but in 'FirstModule' found function template 'foo' with 1st template parameter not being a template parameter pack}}
#endif

#if defined(FIRST)
struct S21 {
  template <template<class> class...> void foo();
  template <template<class> class> void foo();
};
#elif defined(SECOND)
struct S21 {
  template <template<class> class> void foo();
  template <template<class> class...> void foo();
};
#else
S21 s21;
// expected-error@second.h:* {{'FunctionTemplate::S21' has different definitions in different modules; first difference is definition in module 'SecondModule' found function template 'foo' with 1st template parameter not being a template parameter pack}}
// expected-note@first.h:* {{but in 'FirstModule' found function template 'foo' with 1st template parameter being a template parameter pack}}
#endif

#if defined(FIRST)
struct S22 {
  template <template<class> class> void foo();
  template <class> void foo();
  template <int> void foo();
};
#elif defined(SECOND)
struct S22 {
  template <class> void foo();
  template <int> void foo();
  template <template<class> class> void foo();
};
#else
S22 s22;
// expected-error@second.h:* {{'FunctionTemplate::S22' has different definitions in different modules; first difference is definition in module 'SecondModule' found function template 'foo' with 1st template parameter being a type template parameter}}
// expected-note@first.h:* {{but in 'FirstModule' found function template 'foo' with 1st template paramter being a template template parameter}}
#endif

#if defined(FIRST)
struct S23 {
  template <class> void foo();
  template <int> void foo();
  template <template<class> class> void foo();
};
#elif defined(SECOND)
struct S23 {
  template <int> void foo();
  template <template<class> class> void foo();
  template <class> void foo();
};
#else
S23 s23;
// expected-error@second.h:* {{'FunctionTemplate::S23' has different definitions in different modules; first difference is definition in module 'SecondModule' found function template 'foo' with 1st template parameter being a non-type template parameter}}
// expected-note@first.h:* {{but in 'FirstModule' found function template 'foo' with 1st template paramter being a type template parameter}}
#endif

#if defined(FIRST)
struct S24 {
  template <int> void foo();
  template <template<class> class> void foo();
  template <class> void foo();
};
#elif defined(SECOND)
struct S24 {
  template <template<class> class> void foo();
  template <class> void foo();
  template <int> void foo();
};
#else
S24 s24;
// expected-error@second.h:* {{'FunctionTemplate::S24' has different definitions in different modules; first difference is definition in module 'SecondModule' found function template 'foo' with 1st template parameter being a template template parameter}}
// expected-note@first.h:* {{but in 'FirstModule' found function template 'foo' with 1st template paramter being a non-type template parameter}}
#endif

#if defined(FIRST)
struct S25 {
  template <int> void foo();
};
#elif defined(SECOND)
struct S25 {
 public:
  template <int> void foo();
};
#else
S25 s25;
// expected-error@second.h:* {{'FunctionTemplate::S25' has different definitions in different modules; first difference is definition in module 'SecondModule' found public access specifier}}
// expected-note@first.h:* {{but in 'FirstModule' found function template}}
#endif

#define DECLS                                           \
  template <int>                                        \
  void nontype1();                                      \
  template <int x>                                      \
  void nontype2();                                      \
  template <int, int>                                   \
  void nontype3();                                      \
  template <int x = 5>                                  \
  void nontype4();                                      \
  template <int... x>                                   \
  void nontype5();                                      \
                                                        \
  template <class>                                      \
  void type1();                                         \
  template <class x>                                    \
  void type2();                                         \
  template <class, class>                               \
  void type3();                                         \
  template <class x = int>                              \
  void type4();                                         \
  template <class... x>                                 \
  void type5();                                         \
                                                        \
  template <template <int> class>                       \
  void template1();                                     \
  template <template <int> class x>                     \
  void template2();                                     \
  template <template <int> class, template <int> class> \
  void template3();                                     \
  template <template <int> class x = U>                 \
  void template4();                                     \
  template <template <int> class... x>                  \
  void template5();

#if defined(FIRST) || defined(SECOND)
template<int>
struct U {};
struct Valid1 {
  DECLS
};
#else
Valid1 v1;
#endif

#if defined(FIRST) || defined(SECOND)
struct Invalid1 {
  DECLS
  ACCESS
};
#else
Invalid1 i1;
// expected-error@second.h:* {{'FunctionTemplate::Invalid1' has different definitions in different modules; first difference is definition in module 'SecondModule' found private access specifier}}
// expected-note@first.h:* {{but in 'FirstModule' found public access specifier}}
#endif
#undef DECLS
}

// Collection of interesting cases below.

// Naive parsing of AST can lead to cycles in processing.  Ensure
// self-references don't trigger an endless cycles of AST node processing.
namespace SelfReference {
#if defined(FIRST)
template <template <int> class T> class Wrapper {};

template <int N> class S {
  S(Wrapper<::SelfReference::S> &Ref) {}
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
}  // namespace SelfReference

namespace FriendFunction {
#if defined(FIRST)
void F(int = 0);
struct S { friend void F(int); };
#elif defined(SECOND)
void F(int);
struct S { friend void F(int); };
#else
S s;
#endif

#if defined(FIRST)
void G(int = 0);
struct T {
  friend void G(int);

  private:
};
#elif defined(SECOND)
void G(int);
struct T {
  friend void G(int);

  public:
};
#else
T t;
// expected-error@second.h:* {{'FriendFunction::T' has different definitions in different modules; first difference is definition in module 'SecondModule' found public access specifier}}
// expected-note@first.h:* {{but in 'FirstModule' found private access specifier}}
#endif
}  // namespace FriendFunction

namespace ImplicitDecl {
#if defined(FIRST)
struct S { };
void S_Constructors() {
  // Trigger creation of implicit contructors
  S foo;
  S bar = foo;
  S baz(bar);
}
#elif defined(SECOND)
struct S { };
#else
S s;
#endif

#if defined(FIRST)
struct T {
  private:
};
void T_Constructors() {
  // Trigger creation of implicit contructors
  T foo;
  T bar = foo;
  T baz(bar);
}
#elif defined(SECOND)
struct T {
  public:
};
#else
T t;
// expected-error@first.h:* {{'ImplicitDecl::T' has different definitions in different modules; first difference is definition in module 'FirstModule' found private access specifier}}
// expected-note@second.h:* {{but in 'SecondModule' found public access specifier}}
#endif

}  // namespace ImplicitDecl

namespace TemplatedClass {
#if defined(FIRST)
template <class>
struct S {};
#elif defined(SECOND)
template <class>
struct S {};
#else
S<int> s;
#endif

#if defined(FIRST)
template <class>
struct T {
  private:
};
#elif defined(SECOND)
template <class>
struct T {
  public:
};
#else
T<int> t;
// expected-error@second.h:* {{'TemplatedClass::T' has different definitions in different modules; first difference is definition in module 'SecondModule' found public access specifier}}
// expected-note@first.h:* {{but in 'FirstModule' found private access specifier}}
#endif
}  // namespace TemplatedClass

namespace TemplateClassWithField {
#if defined(FIRST)
template <class A>
struct S {
  A a;
};
#elif defined(SECOND)
template <class A>
struct S {
  A a;
};
#else
S<int> s;
#endif

#if defined(FIRST)
template <class A>
struct T {
  A a;

  private:
};
#elif defined(SECOND)
template <class A>
struct T {
  A a;

  public:
};
#else
T<int> t;
// expected-error@second.h:* {{'TemplateClassWithField::T' has different definitions in different modules; first difference is definition in module 'SecondModule' found public access specifier}}
// expected-note@first.h:* {{but in 'FirstModule' found private access specifier}}
#endif
}  // namespace TemplateClassWithField

namespace TemplateClassWithTemplateField {
#if defined(FIRST)
template <class A>
class WrapperS;
template <class A>
struct S {
  WrapperS<A> a;
};
#elif defined(SECOND)
template <class A>
class WrapperS;
template <class A>
struct S {
  WrapperS<A> a;
};
#else
template <class A>
class WrapperS{};
S<int> s;
#endif

#if defined(FIRST)
template <class A>
class WrapperT;
template <class A>
struct T {
  WrapperT<A> a;

  public:
};
#elif defined(SECOND)
template <class A>
class WrapperT;
template <class A>
struct T {
  WrapperT<A> a;

  private:
};
#else
template <class A>
class WrapperT{};
T<int> t;
// expected-error@second.h:* {{'TemplateClassWithTemplateField::T' has different definitions in different modules; first difference is definition in module 'SecondModule' found private access specifier}}
// expected-note@first.h:* {{but in 'FirstModule' found public access specifier}}
#endif
}  // namespace TemplateClassWithTemplateField

namespace EnumWithForwardDeclaration {
#if defined(FIRST)
enum E : int;
struct S {
  void get(E) {}
};
#elif defined(SECOND)
enum E : int { A, B };
struct S {
  void get(E) {}
};
#else
S s;
#endif

#if defined(FIRST)
struct T {
  void get(E) {}
  public:
};
#elif defined(SECOND)
struct T {
  void get(E) {}
  private:
};
#else
T t;
// expected-error@second.h:* {{'EnumWithForwardDeclaration::T' has different definitions in different modules; first difference is definition in module 'SecondModule' found private access specifier}}
// expected-note@first.h:* {{but in 'FirstModule' found public access specifier}}
#endif
}  // namespace EnumWithForwardDeclaration

namespace StructWithForwardDeclaration {
#if defined(FIRST)
struct P {};
struct S {
  struct P *ptr;
};
#elif defined(SECOND)
struct S {
  struct P *ptr;
};
#else
S s;
#endif

#if defined(FIRST)
struct Q {};
struct T {
  struct Q *ptr;
  public:
};
#elif defined(SECOND)
struct T {
  struct Q *ptr;
  private:
};
#else
T t;
// expected-error@second.h:* {{'StructWithForwardDeclaration::T' has different definitions in different modules; first difference is definition in module 'SecondModule' found private access specifier}}
// expected-note@first.h:* {{but in 'FirstModule' found public access specifier}}
#endif
}  // namespace StructWithForwardDeclaration

namespace StructWithForwardDeclarationNoDefinition {
#if defined(FIRST)
struct P;
struct S {
  struct P *ptr;
};
#elif defined(SECOND)
struct S {
  struct P *ptr;
};
#else
S s;
#endif

#if defined(FIRST)
struct Q;
struct T {
  struct Q *ptr;

  public:
};
#elif defined(SECOND)
struct T {
  struct Q *ptr;

  private:
};
#else
T t;
// expected-error@second.h:* {{'StructWithForwardDeclarationNoDefinition::T' has different definitions in different modules; first difference is definition in module 'SecondModule' found private access specifier}}
// expected-note@first.h:* {{but in 'FirstModule' found public access specifier}}
#endif
}  // namespace StructWithForwardDeclarationNoDefinition

namespace LateParsedDefaultArgument {
#if defined(FIRST)
template <typename T>
struct S {
  struct R {
    void foo(T x = 0) {}
  };
};
#elif defined(SECOND)
#else
void run() {
  S<int>::R().foo();
}
#endif
}  // namespace LateParsedDefaultArgument

namespace LateParsedDefaultArgument {
#if defined(FIRST)
template <typename alpha> struct Bravo {
  void charlie(bool delta = false) {}
};
typedef Bravo<char> echo;
echo foxtrot;

Bravo<char> golf;
#elif defined(SECOND)
#else
#endif
}  // LateParsedDefaultArgument

namespace DifferentParameterNameInTemplate {
#if defined(FIRST) || defined(SECOND)
template <typename T>
struct S {
  typedef T Type;

  static void Run(const Type *name_one);
};

template <typename T>
void S<T>::Run(const T *name_two) {}

template <typename T>
struct Foo {
  ~Foo() { Handler::Run(nullptr); }
  Foo() {}

  class Handler : public S<T> {};

  void Get(typename Handler::Type *x = nullptr) {}
  void Add() { Handler::Run(nullptr); }
};
#endif

#if defined(FIRST)
struct Beta;

struct Alpha {
  Alpha();
  void Go() { betas.Get(); }
  Foo<Beta> betas;
};

#elif defined(SECOND)
struct Beta {};

struct BetaHelper {
  void add_Beta() { betas.Add(); }
  Foo<Beta> betas;
};

#else
Alpha::Alpha() {}
#endif
}  // DifferentParameterNameInTemplate

namespace ParameterTest {
#if defined(FIRST)
class X {};
template <typename G>
class S {
  public:
   typedef G Type;
   static inline G *Foo(const G *a, int * = nullptr);
};

template<typename G>
G* S<G>::Foo(const G* aaaa, int*) {}
#elif defined(SECOND)
template <typename G>
class S {
  public:
   typedef G Type;
   static inline G *Foo(const G *a, int * = nullptr);
};

template<typename G>
G* S<G>::Foo(const G* asdf, int*) {}
#else
S<X> s;
#endif
}  // ParameterTest

namespace MultipleTypedefs {
#if defined(FIRST)
typedef int B1;
typedef B1 A1;
struct S1 {
  A1 x;
};
#elif defined(SECOND)
typedef int A1;
struct S1 {
  A1 x;
};
#else
S1 s1;
#endif

#if defined(FIRST)
struct T2 { int x; };
typedef T2 B2;
typedef B2 A2;
struct S2 {
  T2 x;
};
#elif defined(SECOND)
struct T2 { int x; };
typedef T2 A2;
struct S2 {
  T2 x;
};
#else
S2 s2;
#endif

#if defined(FIRST)
using A3 = const int;
using B3 = volatile A3;
struct S3 {
  B3 x = 1;
};
#elif defined(SECOND)
using A3 = volatile const int;
using B3 = A3;
struct S3 {
  B3 x = 1;
};
#else
S3 s3;
#endif

#if defined(FIRST)
using A4 = int;
using B4 = A4;
struct S4 {
  B4 x;
};
#elif defined(SECOND)
using A4 = int;
using B4 = ::MultipleTypedefs::A4;
struct S4 {
  B4 x;
};
#else
S4 s4;
#endif

#if defined(FIRST)
using A5 = int;
using B5 = MultipleTypedefs::A5;
struct S5 {
  B5 x;
};
#elif defined(SECOND)
using A5 = int;
using B5 = ::MultipleTypedefs::A5;
struct S5 {
  B5 x;
};
#else
S5 s5;
#endif
}  // MultipleTypedefs

namespace DefaultArguments {
#if defined(FIRST)
template <typename T>
struct S {
  struct R {
    void foo(T x = 0);
  };
};
#elif defined(SECOND)
template <typename T>
struct S {
  struct R {
    void foo(T x = 1);
  };
};
#else
void run() {
  S<int>::R().foo();
}
// expected-error@second.h:* {{'DefaultArguments::S::R' has different definitions in different modules; first difference is definition in module 'SecondModule' found method 'foo' with 1st parameter with a default argument}}
// expected-note@first.h:* {{but in 'FirstModule' found method 'foo' with 1st parameter with a different default argument}}
#endif

#if defined(FIRST)
template <typename alpha> struct Bravo {
  void charlie(bool delta = false);
};
typedef Bravo<char> echo;
echo foxtrot;
#elif defined(SECOND)
template <typename alpha> struct Bravo {
  void charlie(bool delta = (false));
};
typedef Bravo<char> echo;
echo foxtrot;
#else
Bravo<char> golf;
// expected-error@second.h:* {{'DefaultArguments::Bravo' has different definitions in different modules; first difference is definition in module 'SecondModule' found method 'charlie' with 1st parameter with a default argument}}
// expected-note@first.h:* {{but in 'FirstModule' found method 'charlie' with 1st parameter with a different default argument}}
#endif
}  // namespace DefaultArguments

namespace FunctionDecl {
#if defined(FIRST)
struct S1 {};
S1 s1a;
#elif defined(SECOND)
struct S1 {};
#else
S1 s1;
#endif

#if defined(FIRST)
struct S2 {
  S2() = default;
};
S2 s2a = S2();
#elif defined(SECOND)
struct S2 {
  S2() = default;
};
#else
S2 s2;
#endif

#if defined(FIRST)
struct S3 {
  S3() = delete;
};
S3* s3c;
#elif defined(SECOND)
struct S3 {
  S3() = delete;
};
#else
S3* s3;
#endif

#if defined(FIRST) || defined(SECOND)
int F1(int x, float y = 2.7) { return 1; }
#else
int I1 = F1(1);
#endif

#if defined(FIRST)
int F2() { return 1; }
#elif defined(SECOND)
double F2() { return 1; }
#else
int I2 = F2();
// expected-error@-1 {{call to 'F2' is ambiguous}}
// expected-note@first.h:* {{candidate function}}
// expected-note@second.h:* {{candidate function}}
#endif

#if defined(FIRST)
int F3(float) { return 1; }
#elif defined(SECOND)
int F3(double) { return 1; }
#else
int I3 = F3(1);
// expected-error@-1 {{call to 'F3' is ambiguous}}
// expected-note@first.h:* {{candidate function}}
// expected-note@second.h:* {{candidate function}}
#endif

#if defined(FIRST)
int F4(int x) { return 1; }
#elif defined(SECOND)
int F4(int y) { return 1; }
#else
int I4 = F4(1);
// expected-error@second.h:* {{'FunctionDecl::F4' has different definitions in different modules; definition in module 'SecondModule' first difference is 1st parameter with name 'y'}}
// expected-note@first.h:* {{but in 'FirstModule' found 1st parameter with name 'x'}}
#endif

#if defined(FIRST)
int F5(int x) { return 1; }
#elif defined(SECOND)
int F5(int x = 1) { return 1; }
#else
int I5 = F6(1);
// expected-error@second.h:* {{'FunctionDecl::F5' has different definitions in different modules; definition in module 'SecondModule' first difference is 1st parameter without a default argument}}
// expected-note@first.h:* {{but in 'FirstModule' found 1st parameter with a default argument}}
#endif

#if defined(FIRST)
int F6(int x = 2) { return 1; }
#elif defined(SECOND)
int F6(int x = 1) { return 1; }
#else
int I6 = F6(1);
// expected-error@second.h:* {{'FunctionDecl::F6' has different definitions in different modules; definition in module 'SecondModule' first difference is 1st parameter with a default argument}}
// expected-note@first.h:* {{but in 'FirstModule' found 1st parameter with a different default argument}}
#endif

using I = int;
#if defined(FIRST)
I F7() { return 0; }
#elif defined(SECOND)
int F7() { return 0; }
#else
int I7 = F7();
// expected-error@second.h:* {{'FunctionDecl::F7' has different definitions in different modules; definition in module 'SecondModule' first difference is return type is 'int'}}
// expected-note@first.h:* {{but in 'FirstModule' found different return type 'FunctionDecl::I' (aka 'int')}}
#endif

#if defined(FIRST)
int F8(int) { return 0; }
#elif defined(SECOND)
int F8(I) { return 0; }
#else
int I8 = F8(1);
// expected-error@second.h:* {{'FunctionDecl::F8' has different definitions in different modules; definition in module 'SecondModule' first difference is 1st parameter with type 'FunctionDecl::I' (aka 'int')}}
// expected-note@first.h:* {{but in 'FirstModule' found 1st parameter with type 'int'}}
#endif

#if defined(FIRST)
int F9(int[1]) { return 0; }
#elif defined(SECOND)
int F9(int[2]) { return 0; }
#else
int I9 = F9(nullptr);
// expected-error@second.h:* {{'FunctionDecl::F9' has different definitions in different modules; definition in module 'SecondModule' first difference is 1st parameter with type 'int *' decayed from 'int [2]'}}
// expected-note@first.h:* {{but in 'FirstModule' found 1st parameter with type 'int *' decayed from 'int [1]'}}
#endif

#if defined(FIRST)
int F10() { return 1; }
#elif defined(SECOND)
int F10() { return 2; }
#else
int I10 = F10();
#endif
// expected-error@second.h:* {{'FunctionDecl::F10' has different definitions in different modules; definition in module 'SecondModule' first difference is function body}}
// expected-note@first.h:* {{but in 'FirstModule' found a different body}}

#if defined(FIRST)
struct S11 {
  template <int> void foo();
};
#elif defined(SECOND)
struct S11 {
  template <int> void foo();
};
template <int> void S11::foo() {}
#else
S11 s11;
#endif

}  // namespace FunctionDecl

namespace DeclTemplateArguments {
#if defined(FIRST)
int foo() { return 1; }
int bar() { return foo(); }
#elif defined(SECOND)
template <class T = int>
int foo() { return 2; }
int bar() { return foo<>(); }
#else
int num = bar();
// expected-error@second.h:* {{'DeclTemplateArguments::bar' has different definitions in different modules; definition in module 'SecondModule' first difference is function body}}
// expected-note@first.h:* {{but in 'FirstModule' found a different body}}
#endif
}

// Keep macros contained to one file.
#ifdef FIRST
#undef FIRST
#endif

#ifdef SECOND
#undef SECOND
#endif

#ifdef ACCESS
#undef ACCESS
#endif
