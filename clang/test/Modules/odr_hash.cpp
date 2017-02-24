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
// expected-error@second.h:* {{'Field::S4' has different definitions in different modules; first difference is definition in module 'SecondModule' found field 'x' with type 'B' (aka 'int')}}
// expected-note@first.h:* {{but in 'FirstModule' found field 'x' with type 'A' (aka 'int')}}

S5 s5;
// expected-error@second.h:* {{'Field::S5' has different definitions in different modules; first difference is definition in module 'SecondModule' found field 'x' with type 'int'}}
// expected-note@first.h:* {{but in 'FirstModule' found field 'x' with type 'A' (aka 'int')}}
#endif


}  // namespace Field

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

// Interesting cases that should not cause errors.  struct S should not error
// while struct T should error at the access specifier mismatch at the end.
namespace AllDecls {
#if defined(FIRST)
typedef int INT;
struct S {
  public:
  private:
  protected:

  static_assert(1 == 1, "Message");
  static_assert(2 == 2);

  int x;
  double y;

  INT z;
};
#elif defined(SECOND)
typedef int INT;
struct S {
  public:
  private:
  protected:

  static_assert(1 == 1, "Message");
  static_assert(2 == 2);

  int x;
  double y;

  INT z;
};
#else
S s;
#endif

#if defined(FIRST)
typedef int INT;
struct T {
  public:
  private:
  protected:

  static_assert(1 == 1, "Message");
  static_assert(2 == 2);

  int x;
  double y;

  INT z;

  private:
};
#elif defined(SECOND)
typedef int INT;
struct T {
  public:
  private:
  protected:

  static_assert(1 == 1, "Message");
  static_assert(2 == 2);

  int x;
  double y;

  INT z;

  public:
};
#else
T t;
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

// Keep macros contained to one file.
#ifdef FIRST
#undef FIRST
#endif
#ifdef SECOND
#undef SECOND
#endif
