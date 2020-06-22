// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++11 -Wno-defaulted-function-deleted

struct DefaultedDefCtor1 {};
struct DefaultedDefCtor2 { DefaultedDefCtor2() = default; };
struct DeletedDefCtor { DeletedDefCtor() = delete; DeletedDefCtor(int); }; // expected-note {{explicitly marked deleted here}}
class PrivateDefCtor { PrivateDefCtor() = default; public: PrivateDefCtor(int); };
struct DeletedDtor { ~DeletedDtor() = delete; }; // expected-note 4{{explicitly marked deleted here}}
class PrivateDtor { ~PrivateDtor() = default; };
class Friend {
  Friend() = default; ~Friend() = default;
  friend struct NotDeleted6c;
  friend struct NotDeleted7i;
  friend struct NotDeleted7j;
  friend struct NotDeleted7k;
};
struct UserProvidedDefCtor { UserProvidedDefCtor() {} };
int n;


// A defaulted default constructor for a class X is defined as deleted if:

// - X is a union-like class that has a variant member with a non-trivial
// default constructor,
union Deleted1a { UserProvidedDefCtor u; }; // expected-note {{default constructor of 'Deleted1a' is implicitly deleted because variant field 'u' has a non-trivial default constructor}}
Deleted1a d1a; // expected-error {{implicitly-deleted default constructor}}
union NotDeleted1a { DefaultedDefCtor1 nu; };
NotDeleted1a nd1a;
union NotDeleted1b { DefaultedDefCtor2 nu; };
NotDeleted1b nd1b;

// - any non-static data member with no brace-or-equal-initializer is of
// reference type,
class Deleted2a {
  Deleted2a() = default;  // expected-note 4{{implicitly deleted here}}
  int &a; // expected-note 4{{because field 'a' of reference type 'int &' would not be initialized}}
};
Deleted2a d2a; // expected-error {{implicitly-deleted default constructor}}
struct Deleted2b {
  int &&b; // expected-note {{default constructor of 'Deleted2b' is implicitly deleted because field 'b' of reference type 'int &&' would not be initialized}}
};
Deleted2b d2b; // expected-error {{deleted default constructor}}
class NotDeleted2a { int &a = n; };
NotDeleted2a nd2a;
class NotDeleted2b { int &a = error; }; // expected-error {{undeclared identifier}}
NotDeleted2b nd2b;
class NotDeleted2c { int &&a = static_cast<int&&>(n); };
NotDeleted2c nd2c;
// Note: this one does not have a deleted default constructor even though the
// implicit default constructor is ill-formed!
class NotDeleted2d { int &&a = 0; }; // expected-error {{reference member 'a' binds to a temporary object}} expected-note {{default member init}}
NotDeleted2d nd2d; // expected-note {{first required here}}

// - any non-variant non-static data member of const qualified type (or array
// thereof) with no brace-or-equal-initializer does not have a user-provided
// default constructor,
class Deleted3a { const int a; }; // expected-note {{because field 'a' of const-qualified type 'const int' would not be initialized}} \
                                     expected-warning {{does not declare any constructor}} \
                                     expected-note {{will never be initialized}}
Deleted3a d3a; // expected-error {{implicitly-deleted default constructor}}
class Deleted3b { const DefaultedDefCtor1 a[42]; }; // expected-note {{because field 'a' of const-qualified type 'const DefaultedDefCtor1 [42]' would not be initialized}}
Deleted3b d3b; // expected-error {{implicitly-deleted default constructor}}
class Deleted3c { const DefaultedDefCtor2 a; }; // expected-note {{because field 'a' of const-qualified type 'const DefaultedDefCtor2' would not be initialized}}
Deleted3c d3c; // expected-error {{implicitly-deleted default constructor}}
class NotDeleted3a { const int a = 0; };
NotDeleted3a nd3a;
class NotDeleted3b { const DefaultedDefCtor1 a[42] = {}; };
NotDeleted3b nd3b;
class NotDeleted3c { const DefaultedDefCtor2 a = DefaultedDefCtor2(); };
NotDeleted3c nd3c;
union NotDeleted3d { const int a; int b; };
NotDeleted3d nd3d;
union NotDeleted3e { const DefaultedDefCtor1 a[42]; int b; };
NotDeleted3e nd3e;
union NotDeleted3f { const DefaultedDefCtor2 a; int b; };
NotDeleted3f nd3f;
struct NotDeleted3g { union { const int a; int b; }; };
NotDeleted3g nd3g;

// - X is a union and all of its variant members are of const-qualified type (or
// array thereof),
union Deleted4a {
  const int a;
  const int b;
  const UserProvidedDefCtor c; // expected-note {{because variant field 'c' has a non-trivial default constructor}}
};
Deleted4a d4a; // expected-error {{implicitly-deleted default constructor}}
union NotDeleted4a { const int a; int b; };
NotDeleted4a nd4a;

// - X is a non-union class and all members of any anonymous union member are of
// const-qualified type (or array thereof),
struct Deleted5a {
  union { const int a; }; // expected-note {{because all data members of an anonymous union member are const-qualified}}
  union { int b; };
};
Deleted5a d5a; // expected-error {{implicitly-deleted default constructor}}
struct NotDeleted5a { union { const int a; int b; }; union { const int c; int d; }; };
NotDeleted5a nd5a;

// - any direct or virtual base class, or non-static data member with no
// brace-or-equal-initializer, has class type M (or array thereof) and either
// M has no default constructor or overload resolution as applied to M's default
// constructor results in an ambiguity or in a function that is deleted or
// inaccessible from the defaulted default constructor, or
struct Deleted6a : Deleted2a {}; // expected-note {{because base class 'Deleted2a' has a deleted default constructor}}
Deleted6a d6a; // expected-error {{implicitly-deleted default constructor}}
struct Deleted6b : virtual Deleted2a {}; // expected-note {{because base class 'Deleted2a' has a deleted default constructor}}
Deleted6b d6b; // expected-error {{implicitly-deleted default constructor}}
struct Deleted6c { Deleted2a a; }; // expected-note {{because field 'a' has a deleted default constructor}}
Deleted6c d6c; // expected-error {{implicitly-deleted default constructor}}
struct Deleted6d { DeletedDefCtor a; }; // expected-note {{because field 'a' has a deleted default constructor}}
Deleted6d d6d; // expected-error {{implicitly-deleted default constructor}}
struct NotDeleted6a { DeletedDefCtor a = 0; };
NotDeleted6a nd6a;
struct Deleted6e { PrivateDefCtor a; }; // expected-note {{because field 'a' has an inaccessible default constructor}}
Deleted6e d6e; // expected-error {{implicitly-deleted default constructor}}
struct NotDeleted6b { PrivateDefCtor a = 0; };
NotDeleted6b nd6b;
struct NotDeleted6c { Friend a; };
NotDeleted6c nd6c;

// - any direct or virtual base class or non-static data member has a type with
// a destructor that is deleted or inaccessible from the defaulted default
// constructor.
struct Deleted7a : DeletedDtor {}; // expected-note {{because base class 'DeletedDtor' has a deleted destructor}}
Deleted7a d7a; // expected-error {{implicitly-deleted default constructor}}
struct Deleted7b : virtual DeletedDtor {}; // expected-note {{because base class 'DeletedDtor' has a deleted destructor}}
Deleted7b d7b; // expected-error {{implicitly-deleted default constructor}}
struct Deleted7c { DeletedDtor a; }; // expected-note {{because field 'a' has a deleted destructor}}
Deleted7c d7c; // expected-error {{implicitly-deleted default constructor}}
struct Deleted7d { DeletedDtor a = {}; }; // expected-note {{because field 'a' has a deleted destructor}}
Deleted7d d7d; // expected-error {{implicitly-deleted default constructor}}
struct Deleted7e : PrivateDtor {}; // expected-note {{base class 'PrivateDtor' has an inaccessible destructor}}
Deleted7e d7e; // expected-error {{implicitly-deleted default constructor}}
struct Deleted7f : virtual PrivateDtor {}; // expected-note {{base class 'PrivateDtor' has an inaccessible destructor}}
Deleted7f d7f; // expected-error {{implicitly-deleted default constructor}}
struct Deleted7g { PrivateDtor a; }; // expected-note {{field 'a' has an inaccessible destructor}}
Deleted7g d7g; // expected-error {{implicitly-deleted default constructor}}
struct Deleted7h { PrivateDtor a = {}; }; // expected-note {{field 'a' has an inaccessible destructor}}
Deleted7h d7h; // expected-error {{implicitly-deleted default constructor}}
struct NotDeleted7i : Friend {};
NotDeleted7i d7i;
struct NotDeleted7j : virtual Friend {};
NotDeleted7j d7j;
struct NotDeleted7k { Friend a; };
NotDeleted7k d7k;


class Trivial { static const int n = 42; };
static_assert(__has_trivial_constructor(Trivial), "Trivial is nontrivial");

// A default constructor is trivial if it is not user-provided and if:
class NonTrivialDefCtor1 { NonTrivialDefCtor1(); };
static_assert(!__has_trivial_constructor(NonTrivialDefCtor1), "NonTrivialDefCtor1 is trivial");

#define ASSERT_NONTRIVIAL_IMPL(Class, Bases, Body) \
  class Class Bases { Body }; \
  static_assert(!__has_trivial_constructor(Class), "");
#define ASSERT_NONTRIVIAL(Class, Bases, Body) \
  ASSERT_NONTRIVIAL_IMPL(Class, Bases, Body) \
  ASSERT_NONTRIVIAL_IMPL(Def ## Class, Bases, Def ## Class() = default; Body) \
  ASSERT_NONTRIVIAL_IMPL(Del ## Class, Bases, Del ## Class() = delete; Body)

// - its class has no virtual functions (10.3) and no virtual base classes (10.1), and
ASSERT_NONTRIVIAL(NonTrivialDefCtor2, , virtual void f();)
ASSERT_NONTRIVIAL(NonTrivialDefCtor3, : virtual Trivial, )

// - no non-static data member of its class has a brace-or-equal-initializer, and
ASSERT_NONTRIVIAL(NonTrivialDefCtor4, , int m = 52;)

// - all the direct base classes of its class have trivial default constructors, and
ASSERT_NONTRIVIAL(NonTrivialDefCtor5, : NonTrivialDefCtor1, )

// - for all the non-static data members of its class that are of class type (or array thereof), each such class
// has a trivial default constructor.
ASSERT_NONTRIVIAL(NonTrivialDefCtor6, , NonTrivialDefCtor1 t;)

// FIXME: No core issue number yet.
// - its parameter-declaration-clause is equivalent to that of an implicit declaration.
struct NonTrivialDefCtor7 {
  NonTrivialDefCtor7(...) = delete;
};
static_assert(!__has_trivial_constructor(NonTrivialDefCtor7), "");
struct NonTrivialDefCtor8 {
  NonTrivialDefCtor8(int = 0) = delete;
};
static_assert(!__has_trivial_constructor(NonTrivialDefCtor8), "");

// Otherwise, the default constructor is non-trivial.

class Trivial2 { Trivial2() = delete; };
static_assert(__has_trivial_constructor(Trivial2), "Trivial2 is trivial");

class Trivial3 { Trivial3() = default; };
static_assert(__has_trivial_constructor(Trivial3), "Trivial3 is trivial");

template<typename T> class Trivial4 { Trivial4() = default; };
static_assert(__has_trivial_constructor(Trivial4<int>), "Trivial4 is trivial");

template<typename T> class Trivial5 { Trivial5() = delete; };
static_assert(__has_trivial_constructor(Trivial5<int>), "Trivial5 is trivial");

namespace PR14558 {
  // Ensure we determine whether an explicitly-defaulted or deleted special
  // member is trivial before we return to parsing the containing class.
  struct A {
    struct B { B() = default; } b;
    struct C { C() = delete; } c;
  };

  static_assert(__has_trivial_constructor(A), "");
  static_assert(__has_trivial_constructor(A::B), "");
}
