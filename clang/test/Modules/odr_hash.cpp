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
}

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
  void A(int x) {}
};
#elif defined(SECOND)
struct S11 {
  void A(int y) {}
};
#else
S11 s11;
// expected-error@second.h:* {{'Method::S11' has different definitions in different modules; first difference is definition in module 'SecondModule' found method 'A' with 1st parameter named 'y'}}
// expected-note@first.h:* {{but in 'FirstModule' found method 'A' with 1st parameter named 'x'}}
#endif

#if defined(FIRST)
struct S12 {
  void A(int x) {}
};
#elif defined(SECOND)
struct S12 {
  void A(int x = 1) {}
};
#else
S12 s12;
// TODO: This should produce an error.
#endif

#if defined(FIRST)
struct S13 {
  void A(int x = 1 + 0) {}
};
#elif defined(SECOND)
struct S13 {
  void A(int x = 1) {}
};
#else
S13 s13;
// TODO: This should produce an error.
#endif

#if defined(FIRST)
struct S14 {
  void A(int x[2]) {}
};
#elif defined(SECOND)
struct S14 {
  void A(int x[3]) {}
};
#else
S14 s14;
// expected-error@second.h:* {{'Method::S14' has different definitions in different modules; first difference is definition in module 'SecondModule' found method 'A' with 1st parameter of type 'int *' decayed from 'int [3]'}}
// expected-note@first.h:* {{but in 'FirstModule' found method 'A' with 1st parameter of type 'int *' decayed from 'int [2]'}}
#endif
}  // namespace Method

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
}  // namespace Using


// Interesting cases that should not cause errors.  struct S should not error
// while struct T should error at the access specifier mismatch at the end.
namespace AllDecls {
#define CREATE_ALL_DECL_STRUCT(NAME, ACCESS)               \
  typedef int INT;                                         \
  struct NAME {                                            \
  public:                                                  \
  private:                                                 \
  protected:                                               \
    static_assert(1 == 1, "Message");                      \
    static_assert(2 == 2);                                 \
                                                           \
    int x;                                                 \
    double y;                                              \
                                                           \
    INT z;                                                 \
                                                           \
    unsigned a : 1;                                        \
    unsigned b : 2 * 2 + 5 / 2;                            \
                                                           \
    mutable int c = sizeof(x + y);                         \
                                                           \
    void method() {}                                       \
    static void static_method() {}                         \
    virtual void virtual_method() {}                       \
    virtual void pure_virtual_method() = 0;                \
    inline void inline_method() {}                         \
    void volatile_method() volatile {}                     \
    void const_method() const {}                           \
                                                           \
    typedef int typedef_int;                               \
    using using_int = int;                                 \
                                                           \
    void method_one_arg(int x) {}                          \
    void method_one_arg_default_argument(int x = 5 + 5) {} \
    void method_decayed_type(int x[5]) {}                  \
                                                           \
    int constant_arr[5];                                   \
                                                           \
    ACCESS:                                                \
  };

#if defined(FIRST)
CREATE_ALL_DECL_STRUCT(S, public)
#elif defined(SECOND)
CREATE_ALL_DECL_STRUCT(S, public)
#else
S *s;
#endif

#if defined(FIRST)
CREATE_ALL_DECL_STRUCT(T, private)
#elif defined(SECOND)
CREATE_ALL_DECL_STRUCT(T, public)
#else
T *t;
// expected-error@second.h:* {{'AllDecls::T' has different definitions in different modules; first difference is definition in module 'SecondModule' found public access specifier}}
// expected-note@first.h:* {{but in 'FirstModule' found private access specifier}}
#endif
}

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

}  // namespace ImplicitDelc

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
}

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
}

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
}

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
}


// Keep macros contained to one file.
#ifdef FIRST
#undef FIRST
#endif
#ifdef SECOND
#undef SECOND
#endif
