// RUN: %clang_cc1 -fsyntax-only -std=c++11 -Wc++98-compat -verify %s
// RUN: %clang_cc1 -fsyntax-only -std=c++14 -Wc++98-compat -verify %s -DCXX14COMPAT
// RUN: %clang_cc1 -fsyntax-only -std=c++17 -Wc++98-compat -verify %s -DCXX14COMPAT -DCXX17COMPAT

namespace std {
  struct type_info;
  using size_t = decltype(sizeof(0)); // expected-warning {{decltype}} expected-warning {{alias}}
  template<typename T> struct initializer_list {
    initializer_list(const T*, size_t);
    const T *p;
    size_t n;
    const T *begin();
    const T *end();
  };
}

template<typename ...T>  // expected-warning {{variadic templates are incompatible with C++98}}
class Variadic1 {};

template<template<typename> class ...T>  // expected-warning {{variadic templates are incompatible with C++98}}
class Variadic2 {};

template<int ...I>  // expected-warning {{variadic templates are incompatible with C++98}}
class Variadic3 {};

alignas(8) int with_alignas; // expected-warning {{'alignas' is incompatible with C++98}}
int with_attribute [[ ]]; // expected-warning {{[[]] attributes are incompatible with C++ standards before C++11}}

void Literals() {
  (void)u8"str"; // expected-warning {{unicode literals are incompatible with C++98}}
  (void)u"str"; // expected-warning {{unicode literals are incompatible with C++98}}
  (void)U"str"; // expected-warning {{unicode literals are incompatible with C++98}}
  (void)u'x'; // expected-warning {{unicode literals are incompatible with C++98}}
  (void)U'x'; // expected-warning {{unicode literals are incompatible with C++98}}

  (void)u8R"X(str)X"; // expected-warning {{raw string literals are incompatible with C++98}}
  (void)uR"X(str)X"; // expected-warning {{raw string literals are incompatible with C++98}}
  (void)UR"X(str)X"; // expected-warning {{raw string literals are incompatible with C++98}}
  (void)R"X(str)X"; // expected-warning {{raw string literals are incompatible with C++98}}
  (void)LR"X(str)X"; // expected-warning {{raw string literals are incompatible with C++98}}
}

template<typename T> struct S {};
namespace TemplateParsing {
  S<::S<void> > s; // expected-warning {{'<::' is treated as digraph '<:' (aka '[') followed by ':' in C++98}}
  S< ::S<void>> t; // expected-warning {{consecutive right angle brackets are incompatible with C++98 (use '> >')}}
}

void Lambda() {
  []{}(); // expected-warning {{lambda expressions are incompatible with C++98}}
  // Don't warn about implicit "-> auto" here.
  [](){}(); // expected-warning {{lambda expressions are incompatible with C++98}}
}

struct Ctor {
  Ctor(int, char);
  Ctor(double, long);
};
struct InitListCtor {
  InitListCtor(std::initializer_list<bool>);
};

int InitList(int i = {}) { // expected-warning {{generalized initializer lists are incompatible with C++98}} \
                           // expected-warning {{scalar initialized from empty initializer list is incompatible with C++98}}
  (void)new int {}; // expected-warning {{generalized initializer lists are incompatible with C++98}} \
                    // expected-warning {{scalar initialized from empty initializer list is incompatible with C++98}}
  (void)int{}; // expected-warning {{generalized initializer lists are incompatible with C++98}} \
               // expected-warning {{scalar initialized from empty initializer list is incompatible with C++98}}
  int x { 0 }; // expected-warning {{generalized initializer lists are incompatible with C++98}}
  S<int> s = {}; // ok, aggregate
  s = {}; // expected-warning {{generalized initializer lists are incompatible with C++98}}
  std::initializer_list<int> xs = { 1, 2, 3 }; // expected-warning {{initialization of initializer_list object is incompatible with C++98}}
  auto ys = { 1, 2, 3 }; // expected-warning {{initialization of initializer_list object is incompatible with C++98}} \
                         // expected-warning {{'auto' type specifier is incompatible with C++98}}
  Ctor c1 = { 1, 2 }; // expected-warning {{constructor call from initializer list is incompatible with C++98}}
  Ctor c2 = { 3.0, 4l }; // expected-warning {{constructor call from initializer list is incompatible with C++98}}
  InitListCtor ilc = { true, false }; // expected-warning {{initialization of initializer_list object is incompatible with C++98}}
  const int &r = { 0 }; // expected-warning {{reference initialized from initializer list is incompatible with C++98}}
  struct { int a; const int &r; } rr = { 0, {0} }; // expected-warning {{reference initialized from initializer list is incompatible with C++98}}
  return { 0 }; // expected-warning {{generalized initializer lists are incompatible with C++98}} expected-warning {{scalar}}
}
struct DelayedDefaultArgumentParseInitList {
  void f(int i = {1}) { // expected-warning {{generalized initializer lists are incompatible with C++98}} expected-warning {{scalar}}
  }
};

int operator""_hello(const char *); // expected-warning {{literal operators are incompatible with C++98}}

enum EnumFixed : int { // expected-warning {{enumeration types with a fixed underlying type are incompatible with C++98}}
};

enum class EnumScoped { // expected-warning {{scoped enumerations are incompatible with C++98}}
};

void Deleted() = delete; // expected-warning {{deleted function definitions are incompatible with C++98}}
struct Defaulted {
  Defaulted() = default; // expected-warning {{defaulted function definitions are incompatible with C++98}}
};

int &&RvalueReference = 0; // expected-warning {{rvalue references are incompatible with C++98}}
struct RefQualifier {
  void f() &; // expected-warning {{reference qualifiers on functions are incompatible with C++98}}
};

auto f() -> int; // expected-warning {{trailing return types are incompatible with C++98}}
#ifdef CXX14COMPAT
auto ff() { return 5; } // expected-warning {{'auto' type specifier is incompatible with C++98}}
// expected-warning@-1 {{return type deduction is incompatible with C++ standards before C++14}}
#endif

void RangeFor() {
  int xs[] = {1, 2, 3};
  for (int &a : xs) { // expected-warning {{range-based for loop is incompatible with C++98}}
  }
  for (auto &b : {1, 2, 3}) {
  // expected-warning@-1 {{range-based for loop is incompatible with C++98}}
  // expected-warning@-2 {{'auto' type specifier is incompatible with C++98}}
  // expected-warning@-3 {{initialization of initializer_list object is incompatible with C++98}}
  // expected-warning@-4 {{reference initialized from initializer list is incompatible with C++98}}
  }
  struct Agg { int a, b; } const &agg = { 1, 2 }; // expected-warning {{reference initialized from initializer list is incompatible with C++98}}
}

struct InClassInit {
  int n = 0; // expected-warning {{default member initializer for non-static data members is incompatible with C++98}}
};

struct OverrideControlBase {
  virtual void f();
  virtual void g();
};
struct OverrideControl final : OverrideControlBase { // expected-warning {{'final' keyword is incompatible with C++98}}
  virtual void f() override; // expected-warning {{'override' keyword is incompatible with C++98}}
  virtual void g() final; // expected-warning {{'final' keyword is incompatible with C++98}}
};

using AliasDecl = int; // expected-warning {{alias declarations are incompatible with C++98}}
template<typename T> using AliasTemplate = T; // expected-warning {{alias declarations are incompatible with C++98}}

inline namespace InlineNS { // expected-warning {{inline namespaces are incompatible with C++98}}
}

auto auto_deduction = 0; // expected-warning {{'auto' type specifier is incompatible with C++98}}
int *p = new auto(0); // expected-warning {{'auto' type specifier is incompatible with C++98}}

const int align_of = alignof(int); // expected-warning {{alignof expressions are incompatible with C++98}}
char16_t c16 = 0; // expected-warning {{'char16_t' type specifier is incompatible with C++98}}
char32_t c32 = 0; // expected-warning {{'char32_t' type specifier is incompatible with C++98}}
constexpr int const_expr = 0; // expected-warning {{'constexpr' specifier is incompatible with C++98}}
decltype(const_expr) decl_type = 0; // expected-warning {{'decltype' type specifier is incompatible with C++98}}
__decltype(const_expr) decl_type2 = 0; // ok
void no_except() noexcept; // expected-warning {{noexcept specifications are incompatible with C++98}}
bool no_except_expr = noexcept(1 + 1); // expected-warning {{noexcept expressions are incompatible with C++98}}
void *null = nullptr; // expected-warning {{'nullptr' is incompatible with C++98}}
static_assert(true, "!"); // expected-warning {{'static_assert' declarations are incompatible with C++98}}

struct InhCtorBase {
  InhCtorBase(int);
};
struct InhCtorDerived : InhCtorBase {
  using InhCtorBase::InhCtorBase; // expected-warning {{inheriting constructors are incompatible with C++98}}
};

struct FriendMember {
  static void MemberFn();
  friend void FriendMember::MemberFn(); // expected-warning {{friend declaration naming a member of the declaring class is incompatible with C++98}}
};

struct DelegCtor {
  DelegCtor(int) : DelegCtor() {} // expected-warning {{delegating constructors are incompatible with C++98}}
  DelegCtor();
};

template<int n = 0> void DefaultFuncTemplateArg(); // expected-warning {{default template arguments for a function template are incompatible with C++98}}

template<typename T> int TemplateFn(T) { return 0; }
void LocalTemplateArg() {
  struct S {};
  TemplateFn(S()); // expected-warning {{local type 'S' as template argument is incompatible with C++98}}
}
struct {} obj_of_unnamed_type; // expected-note {{here}}
int UnnamedTemplateArg = TemplateFn(obj_of_unnamed_type); // expected-warning {{unnamed type as template argument is incompatible with C++98}}

// FIXME: We do not implement C++98 compatibility warnings for the C++17
// template argument evaluation rules.
#ifndef CXX17COMPAT
namespace RedundantParensInAddressTemplateParam {
  int n;
  template<int*p> struct S {};
  S<(&n)> s; // expected-warning {{redundant parentheses surrounding address non-type template argument are incompatible with C++98}}
  S<(((&n)))> t; // expected-warning {{redundant parentheses surrounding address non-type template argument are incompatible with C++98}}
}
#endif

namespace TemplateSpecOutOfScopeNs {
  template<typename T> struct S {};
}
template<> struct TemplateSpecOutOfScopeNs::S<char> {};

struct Typename {
  template<typename T> struct Inner {};
};
typename ::Typename TypenameOutsideTemplate(); // expected-warning {{use of 'typename' outside of a template is incompatible with C++98}}
Typename::template Inner<int> TemplateOutsideTemplate(); // expected-warning {{use of 'template' keyword outside of a template is incompatible with C++98}}

struct TrivialButNonPOD {
  int f(int);
private:
  int k;
};
void Ellipsis(int n, ...);
void TrivialButNonPODThroughEllipsis() {
  Ellipsis(1, TrivialButNonPOD()); // expected-warning {{passing object of trivial but non-POD type 'TrivialButNonPOD' through variadic function is incompatible with C++98}}
}

struct HasExplicitConversion {
  explicit operator bool(); // expected-warning {{explicit conversion functions are incompatible with C++98}}
};

struct Struct {};
enum Enum { enum_val = 0 };
struct BadFriends {
  friend enum ::Enum; // expected-warning {{elaborated enum specifier cannot be declared as a friend}}
                      // expected-note@-1 {{remove 'enum' to befriend an enum}}
  friend int; // expected-warning {{non-class friend type 'int' is incompatible with C++98}}
  friend Struct; // expected-warning {{befriending 'Struct' without 'struct' keyword is incompatible with C++98}}
};

int n = {}; // expected-warning {{scalar initialized from empty initializer list is incompatible with C++98}}

class PrivateMember {
  struct ImPrivate {};
};
template<typename T> typename T::ImPrivate SFINAEAccessControl(T t) { // expected-warning {{substitution failure due to access control is incompatible with C++98}}
  return typename T::ImPrivate();
}
int SFINAEAccessControl(...) { return 0; }
int CheckSFINAEAccessControl = SFINAEAccessControl(PrivateMember()); // expected-note {{while substituting deduced template arguments into function template 'SFINAEAccessControl' [with T = PrivateMember]}}

namespace UnionOrAnonStructMembers {
  struct NonTrivCtor {
    NonTrivCtor(); // expected-note 2{{user-provided default constructor}}
  };
  struct NonTrivCopy {
    NonTrivCopy(const NonTrivCopy&); // expected-note 2{{user-provided copy constructor}}
  };
  struct NonTrivDtor {
    ~NonTrivDtor(); // expected-note 2{{user-provided destructor}}
  };
  union BadUnion {
    NonTrivCtor ntc; // expected-warning {{union member 'ntc' with a non-trivial default constructor is incompatible with C++98}}
    NonTrivCopy ntcp; // expected-warning {{union member 'ntcp' with a non-trivial copy constructor is incompatible with C++98}}
    NonTrivDtor ntd; // expected-warning {{union member 'ntd' with a non-trivial destructor is incompatible with C++98}}
  };
  struct Wrap {
    struct {
      NonTrivCtor ntc; // expected-warning {{anonymous struct member 'ntc' with a non-trivial default constructor is incompatible with C++98}}
      NonTrivCopy ntcp; // expected-warning {{anonymous struct member 'ntcp' with a non-trivial copy constructor is incompatible with C++98}}
      NonTrivDtor ntd; // expected-warning {{anonymous struct member 'ntd' with a non-trivial destructor is incompatible with C++98}}
    };
  };
  union WithStaticDataMember {
    static constexpr double d = 0.0; // expected-warning {{static data member 'd' in union is incompatible with C++98}} expected-warning {{'constexpr' specifier is incompatible with C++98}}
    static const int n = 0; // expected-warning {{static data member 'n' in union is incompatible with C++98}}
    static int k; // expected-warning {{static data member 'k' in union is incompatible with C++98}}
  };
}

int EnumNNS = Enum::enum_val; // expected-warning {{enumeration type in nested name specifier is incompatible with C++98}}
template<typename T> void EnumNNSFn() {
  int k = T::enum_val; // expected-warning {{enumeration type in nested name specifier is incompatible with C++98}}
};
template void EnumNNSFn<Enum>(); // expected-note {{in instantiation}}

void JumpDiagnostics(int n) {
  goto DirectJump; // expected-warning {{jump from this goto statement to its label is incompatible with C++98}}
  TrivialButNonPOD tnp1; // expected-note {{jump bypasses initialization of non-POD variable}}

DirectJump:
  void *Table[] = {&&DirectJump, &&Later};
  goto *Table[n]; // expected-warning {{jump from this indirect goto statement to one of its possible targets is incompatible with C++98}}

  TrivialButNonPOD tnp2; // expected-note {{jump bypasses initialization of non-POD variable}}
Later: // expected-note {{possible target of indirect goto statement}}
  switch (n) {
    TrivialButNonPOD tnp3; // expected-note {{jump bypasses initialization of non-POD variable}}
  default: // expected-warning {{jump from switch statement to this case label is incompatible with C++98}}
    return;
  }
}

namespace UnevaluatedMemberAccess {
  struct S {
    int n;
    int f() { return sizeof(S::n); } // ok
  };
  int k = sizeof(S::n); // expected-warning {{use of non-static data member 'n' in an unevaluated context is incompatible with C++98}}
  const std::type_info &ti = typeid(S::n); // expected-warning {{use of non-static data member 'n' in an unevaluated context is incompatible with C++98}}
}

namespace LiteralUCNs {
  char c1 = '\u001e'; // expected-warning {{universal character name referring to a control character is incompatible with C++98}}
  wchar_t c2 = L'\u0041'; // expected-warning {{specifying character 'A' with a universal character name is incompatible with C++98}}
  const char *s1 = "foo\u0031"; // expected-warning {{specifying character '1' with a universal character name is incompatible with C++98}}
  const wchar_t *s2 = L"bar\u0085"; // expected-warning {{universal character name referring to a control character is incompatible with C++98}}
}

// FIXME: We do not implement C++98 compatibility warnings for the C++17
// template argument evaluation rules.
#ifndef CXX17COMPAT
namespace NonTypeTemplateArgs {
  template<typename T, T v> struct S {};
  const int k = 5; // expected-note {{here}}
  static void f() {} // expected-note {{here}}
  S<const int&, k> s1; // expected-warning {{non-type template argument referring to object 'k' with internal linkage is incompatible with C++98}}
  S<void(&)(), f> s2; // expected-warning {{non-type template argument referring to function 'f' with internal linkage is incompatible with C++98}}
}

namespace NullPointerTemplateArg {
  struct A {};
  template<int*> struct X {};
  template<int A::*> struct Y {};
  X<(int*)0> x; // expected-warning {{use of null pointer as non-type template argument is incompatible with C++98}}
  Y<(int A::*)0> y; // expected-warning {{use of null pointer as non-type template argument is incompatible with C++98}}
}
#endif

namespace PR13480 {
  struct basic_iterator {
    basic_iterator(const basic_iterator &it) {} // expected-note {{because type 'PR13480::basic_iterator' has a user-provided copy constructor}}
    basic_iterator(basic_iterator &it) {}
  };

  union test {
    basic_iterator it; // expected-warning {{union member 'it' with a non-trivial copy constructor is incompatible with C++98}}
  };
}

namespace AssignOpUnion {
  struct a {
    void operator=(const a &it) {} // expected-note {{because type 'AssignOpUnion::a' has a user-provided copy assignment operator}}
    void operator=(a &it) {}
  };

  struct b {
    void operator=(const b &it) {} // expected-note {{because type 'AssignOpUnion::b' has a user-provided copy assignment operator}}
  };

  union test1 {
    a x; // expected-warning {{union member 'x' with a non-trivial copy assignment operator is incompatible with C++98}}
    b y; // expected-warning {{union member 'y' with a non-trivial copy assignment operator is incompatible with C++98}}
  };
}

namespace rdar11736429 {
  struct X { // expected-note {{because type 'rdar11736429::X' has no default constructor}}
    X(const X&) = delete; // expected-warning{{deleted function definitions are incompatible with C++98}} \
    // expected-note {{implicit default constructor suppressed by user-declared constructor}}
  };

  union S {
    X x; // expected-warning{{union member 'x' with a non-trivial default constructor is incompatible with C++98}}
  };
}

template<typename T> T var = T(10);
#ifdef CXX14COMPAT
// expected-warning@-2 {{variable templates are incompatible with C++ standards before C++14}}
#else
// expected-warning@-4 {{variable templates are a C++14 extension}}
#endif

// No diagnostic for specializations of variable templates; we will have
// diagnosed the primary template.
template<typename T> T* var<T*> = new T();
template<> int var<int> = 10;
template char var<char>;
float fvar = var<float>;

class A {
  template<typename T> static T var = T(10);
#ifdef CXX14COMPAT
// expected-warning@-2 {{variable templates are incompatible with C++ standards before C++14}}
#else
// expected-warning@-4 {{variable templates are a C++14 extension}}
#endif

  template<typename T> static T* var<T*> = new T();
};

struct B {  template<typename T> static T v; };
#ifdef CXX14COMPAT
// expected-warning@-2 {{variable templates are incompatible with C++ standards before C++14}}
#else
// expected-warning@-4 {{variable templates are a C++14 extension}}
#endif

template<typename T> T B::v = T();
#ifdef CXX14COMPAT
// expected-warning@-2 {{variable templates are incompatible with C++ standards before C++14}}
#else
// expected-warning@-4 {{variable templates are a C++14 extension}}
#endif

template<typename T> T* B::v<T*> = new T();
template<> int B::v<int> = 10;
template char B::v<char>;
float fsvar = B::v<float>;

#ifdef CXX14COMPAT
int digit_seps = 123'456; // expected-warning {{digit separators are incompatible with C++ standards before C++14}}
#endif

#ifdef CXX17COMPAT
template<class T> struct CTAD {};
void ctad_test() {
  CTAD<int> s;
  CTAD t = s; // expected-warning {{class template argument deduction is incompatible with C++ standards before C++17}}
}
#endif
