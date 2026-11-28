// RUN: %clang_cc1 -fsyntax-only -std=c++11 -Wc++98-compat -Wno-c++14-extensions                   -verify=expected,compat,until-cxx14 %s
// RUN: %clang_cc1 -fsyntax-only -std=c++14 -Wc++98-compat -Wno-pre-c++14-compat-pedantic          -verify=expected,compat,until-cxx14 %s
// RUN: %clang_cc1 -fsyntax-only -std=c++17 -Wc++98-compat -Wno-pre-c++14-compat-pedantic          -verify=expected,compat %s
// RUN: %clang_cc1 -fsyntax-only -std=c++20 -Wc++98-compat -Wno-pre-c++14-compat-pedantic          -verify=expected,compat %s
// RUN: %clang_cc1 -fsyntax-only -std=c++23 -Wc++98-compat -Wno-pre-c++14-compat-pedantic          -verify=expected,compat %s
// RUN: %clang_cc1 -fsyntax-only -std=c++26 -Wc++98-compat -Wno-pre-c++14-compat-pedantic          -verify=expected,compat %s

// RUN: %clang_cc1 -fsyntax-only -std=c++11 -Wc++98-compat-pedantic -Wno-c++14-extensions          -verify=expected,compat,pedantic,pedantic-until-cxx14,pedantic-cxx11,until-cxx14 %s
// RUN: %clang_cc1 -fsyntax-only -std=c++14 -Wc++98-compat-pedantic -Wno-pre-c++14-compat-pedantic -verify=expected,compat,pedantic,pedantic-until-cxx14,until-cxx14 %s
// RUN: %clang_cc1 -fsyntax-only -std=c++17 -Wc++98-compat-pedantic -Wno-pre-c++14-compat-pedantic -verify=expected,compat,pedantic %s
// RUN: %clang_cc1 -fsyntax-only -std=c++20 -Wc++98-compat-pedantic -Wno-pre-c++14-compat-pedantic -verify=expected,compat,pedantic %s
// RUN: %clang_cc1 -fsyntax-only -std=c++23 -Wc++98-compat-pedantic -Wno-pre-c++14-compat-pedantic -verify=expected,compat,pedantic %s
// RUN: %clang_cc1 -fsyntax-only -std=c++26 -Wc++98-compat-pedantic -Wno-pre-c++14-compat-pedantic -verify=expected,compat,pedantic %s

// Make sure that non-pedantic warnings are actually errors in C++03
// RUN: %clang_cc1 -fsyntax-only -std=c++03                                                        -verify=expected,cxx03 %s

// FIXME: This test shows a few bugs:
// - Most places matching the regex 'compat-warning[^\n]*\n^((?!cxx03-error)[^\n])*\n': The warning should be pedantic
// - The 'pedantic-until-cxx14' 'until-cxx14' verify flags: The warning should be in all standards modes

namespace std {
  struct type_info;
  using size_t = __decltype(sizeof(0)); // pedantic-warning {{alias}}
  template<typename T> struct initializer_list {
    initializer_list(const T*, size_t);
    const T *p;
    size_t n;
    const T *begin();
    const T *end();
  };
}

void test_other_auto_spellings() {
  __auto_type x = 0; // Ok
  decltype(auto) y = 0; // compat-warning {{'decltype' type specifier is incompatible with C++98}} \
                        // cxx03-error 2 {{expected}} cxx03-error {{use of undeclared identifier}} \
                        // cxx03-warning {{'auto' type specifier is a C++11 extension}}
}

template<typename ...T>  // pedantic-warning {{variadic templates are incompatible with C++98}}
class Variadic1 {};

template<template<typename> class ...T>  // pedantic-warning {{variadic templates are incompatible with C++98}}
class Variadic2 {};

template<int ...I>  // pedantic-warning {{variadic templates are incompatible with C++98}}
class Variadic3 {};

alignas(8) int with_alignas; // compat-warning {{'alignas' is incompatible with C++98}} \
                             // cxx03-error {{a type specifier}} cxx03-error {{expected}}

void Literals() {
  (void)u8"str"; // compat-warning {{unicode literals are incompatible with C++98}} \
                 // cxx03-error {{undeclared identifier}}
  (void)u"str";  // compat-warning {{unicode literals are incompatible with C++98}} \
                 // cxx03-error {{undeclared identifier}}
  (void)U"str";  // compat-warning {{unicode literals are incompatible with C++98}} \
                 // cxx03-error {{undeclared identifier}}
  (void)u'x';    // compat-warning {{unicode literals are incompatible with C++98}} \
                 // cxx03-error {{undeclared identifier}}
  (void)U'x';    // compat-warning {{unicode literals are incompatible with C++98}} \
                 // cxx03-error {{undeclared identifier}}

  (void)u8R"X(str)X"; // compat-warning {{raw string literals are incompatible with C++98}} \
                      // cxx03-error {{undeclared identifier}}
  (void)uR"X(str)X";  // compat-warning {{raw string literals are incompatible with C++98}} \
                      // cxx03-error {{undeclared identifier}}
  (void)UR"X(str)X";  // compat-warning {{raw string literals are incompatible with C++98}} \
                      // cxx03-error {{undeclared identifier}}
  (void)R"X(str)X";   // compat-warning {{raw string literals are incompatible with C++98}} \
                      // cxx03-error {{undeclared identifier}}
  (void)LR"X(str)X";  // compat-warning {{raw string literals are incompatible with C++98}} \
                      // cxx03-error {{undeclared identifier}}
}

template<typename T> struct S {};
namespace TemplateParsing {
  S<::S<void> > s; // compat-warning {{'<::' is treated as digraph '<:' (aka '[') followed by ':' in C++98}} \
                   // cxx03-error {{found '<::' after a template name which forms the digraph '<:' (aka '[') and a ':', did you mean '< ::'?}}
  S< ::S<void>> t; // compat-warning {{consecutive right angle brackets are incompatible with C++98 (use '> >')}} \
                   // cxx03-error {{a space is required between consecutive right angle brackets (use '> >')}}
}

void Lambda() {
  []{}(); // compat-warning {{lambda expressions are incompatible with C++98}} \
          // cxx03-warning {{lambdas are a C++11 extension}}
  // Don't warn about implicit "-> auto" here.
  [](){}(); // compat-warning {{lambda expressions are incompatible with C++98}} \
            // cxx03-warning {{lambdas are a C++11 extension}}
}

struct Ctor {
  Ctor(int, char);
  Ctor(double, long);
};
struct InitListCtor {
  InitListCtor(std::initializer_list<bool>);
};

int InitList(int i = {}) { // compat-warning {{generalized initializer lists are incompatible with C++98}} \
                           // compat-warning {{scalar initialized from empty initializer list is incompatible with C++98}} \
                           // cxx03-error {{expected expression}}
  (void)new int {}; // compat-warning {{generalized initializer lists are incompatible with C++98}} \
                    // compat-warning {{scalar initialized from empty initializer list is incompatible with C++98}} \
                    // cxx03-error {{expected}}
  (void)int{}; // compat-warning {{generalized initializer lists are incompatible with C++98}} \
               // compat-warning {{scalar initialized from empty initializer list is incompatible with C++98}} \
               // cxx03-error {{expected}}
  int x { 0 }; // compat-warning {{generalized initializer lists are incompatible with C++98}} \
               // cxx03-error {{expected}}
  S<int> s = {}; // ok, aggregate
  s = {}; // compat-warning {{generalized initializer lists are incompatible with C++98}} \
          // cxx03-error {{expected expression}}
  std::initializer_list<int> xs = { 1, 2, 3 }; // compat-warning {{initialization of initializer_list object is incompatible with C++98}} \
                                               // cxx03-error {{cannot be initialized}}

  auto ys = { 1, 2, 3 }; // compat-warning {{initialization of initializer_list object is incompatible with C++98}} \
                         // cxx03-error {{cannot be initialized}} \
                         // compat-warning {{'auto' type specifier is incompatible with C++98}} \
                         // cxx03-warning {{'auto' type specifier is a C++11 extension}}

  Ctor c1 = { 1, 2 }; // compat-warning {{constructor call from initializer list is incompatible with C++98}} \
                      // cxx03-error {{cannot be initialized}}

  Ctor c2 = { 3.0, 4l }; // compat-warning {{constructor call from initializer list is incompatible with C++98}} \
                         // cxx03-error {{cannot be initialized}}

  InitListCtor ilc = { true, false }; // compat-warning {{initialization of initializer_list object is incompatible with C++98}} \
                                      // cxx03-error {{cannot be initialized}}

  const int &r = { 0 }; // compat-warning {{reference initialized from initializer list is incompatible with C++98}} \
                        // cxx03-error {{cannot bind}}
  struct { int a; const int &r; } rr = { 0, {0} }; // compat-warning {{reference initialized from initializer list is incompatible with C++98}} \
                                                   // cxx03-error {{initialization of non-aggregate type}}

  return { 0 }; // compat-warning {{generalized initializer lists are incompatible with C++98}} \
                // expected-warning {{scalar}} \
                // cxx03-warning {{generalized initializer lists are a C++11 extension}}
}
struct DelayedDefaultArgumentParseInitList {
  void f(int i = {1}) { // compat-warning {{generalized initializer lists are incompatible with C++98}} compat-warning {{scalar}} \
                        // cxx03-error {{expected expression}}
  }
};

int operator""_hello(const char *); // compat-warning {{literal operators are incompatible with C++98}} \
                                    // cxx03-error {{expected a type}}

enum class EnumScoped { // pedantic-warning {{scoped enumerations are incompatible with C++98}}
};

void Deleted() = delete; // pedantic-warning {{deleted function definitions are incompatible with C++98}}
struct Defaulted {
  Defaulted() = default; // pedantic-warning {{defaulted function definitions are incompatible with C++98}}
};

int &&RvalueReference = 0; // pedantic-warning {{rvalue references are incompatible with C++98}}
struct RefQualifier {
  void f() &; // pedantic-warning {{reference qualifiers on functions are incompatible with C++98}}
};

auto f() -> int; // compat-warning {{trailing return types are incompatible with C++98}} \
                 // cxx03-error {{expected function body after function declarator}} \
                 // cxx03-warning {{'auto' type specifier is a C++11 extension}}

void RangeFor() {
  int xs[] = {1, 2, 3};
  for (int &a : xs) { // pedantic-warning {{range-based for loop is incompatible with C++98}}
  }
  for (auto &b : {1, 2, 3}) {
  // pedantic-warning@-1 {{range-based for loop is incompatible with C++98}}
  // compat-warning@-2 {{'auto' type specifier is incompatible with C++98}}
  // compat-warning@-3 {{initialization of initializer_list object is incompatible with C++98}}
  // cxx03-error@-4 {{reference to type 'std::initializer_list<int>' cannot bind to an initializer list}}
  // compat-warning@-5 {{reference initialized from initializer list is incompatible with C++98}}
  // cxx03-error@-6 {{invalid range expression}}
  // cxx03-warning@-7 {{'auto' type specifier is a C++11 extension}}
  }
  struct Agg { int a, b; } const &agg = { 1, 2 }; // compat-warning {{reference initialized from initializer list is incompatible with C++98}} \
                                                  // cxx03-error {{cannot bind to an initializer list}}
}

struct InClassInit {
  int n = 0; // pedantic-warning {{default member initializer for non-static data member is incompatible with C++98}}
};

struct OverrideControlBase {
  virtual void f();
  virtual void g();
};
struct OverrideControl final : OverrideControlBase { // pedantic-warning {{'final' keyword is incompatible with C++98}}
  virtual void f() override; // pedantic-warning {{'override' keyword is incompatible with C++98}}
  virtual void g() final; // pedantic-warning {{'final' keyword is incompatible with C++98}}
};

using AliasDecl = int; // pedantic-warning {{alias declarations are incompatible with C++98}}
template<typename T> using AliasTemplate = T; // pedantic-warning {{alias declarations are incompatible with C++98}}

inline namespace InlineNS { // pedantic-warning {{inline namespaces are incompatible with C++98}}
}

auto auto_deduction = 0; // compat-warning {{'auto' type specifier is incompatible with C++98}} \
                         // cxx03-warning {{'auto' type specifier is a C++11 extension}}
int *p = new auto(0); // compat-warning {{'auto' type specifier is incompatible with C++98}} \
                      // cxx03-warning 2 {{'auto' type specifier is a C++11 extension}}

const int align_of = alignof(int); // compat-warning {{alignof expressions are incompatible with C++98}} \
                                   // cxx03-error {{expected}}

char16_t c16 = 0; // compat-warning {{'char16_t' type specifier is incompatible with C++98}} \
                  // cxx03-error {{unknown type name}}

char32_t c32 = 0; // compat-warning {{'char32_t' type specifier is incompatible with C++98}} \
                  // cxx03-error {{unknown type name}}

constexpr int const_expr = 0; // compat-warning {{'constexpr' specifier is incompatible with C++98}} \
                              // cxx03-error {{unknown type name}}

decltype(const_expr) decl_type = 0; // compat-warning {{'decltype' type specifier is incompatible with C++98}} \
                                    // cxx03-error {{unknown type name}} cxx03-error {{expected}}

__decltype(const_expr) decl_type2 = 0; // ok

void no_except() noexcept; // compat-warning {{noexcept specifications are incompatible with C++98}} \
                           // cxx03-error {{expected}}

bool no_except_expr = noexcept(1 + 1); // compat-warning {{noexcept expressions are incompatible with C++98}} \
                                       // cxx03-error {{undeclared identifier 'noexcept'}}

void *null = nullptr; // compat-warning {{'nullptr' is incompatible with C++98}} \
                      // cxx03-error {{undeclared identifier 'nullptr'}}

static_assert(true, "!"); // compat-warning {{'static_assert' declarations are incompatible with C++98}} \
                          // cxx03-error {{a type specifier}}

struct InhCtorBase {
  InhCtorBase(int);
};
struct InhCtorDerived : InhCtorBase {
  using InhCtorBase::InhCtorBase; // compat-warning {{inheriting constructors are incompatible with C++98}} \
                                  // cxx03-error {{using declaration cannot refer to a constructor}}
};

struct FriendMember {
  static void MemberFn();
  friend void FriendMember::MemberFn(); // compat-warning {{friend declaration naming a member of the declaring class is incompatible with C++98}} \
                                        // cxx03-error {{friends cannot be members of the declaring class}}
};

struct DelegCtor {
  DelegCtor(int) : DelegCtor() {} // compat-warning {{delegating constructors are incompatible with C++98}} \
                                  // cxx03-error {{delegating constructors are permitted only in C++11}}
  DelegCtor();
};

template<int n = 0> void DefaultFuncTemplateArg(); // pedantic-warning {{default template arguments for a function template are incompatible with C++98}}

template<typename T> int TemplateFn(T) { return 0; }
void LocalTemplateArg() {
  struct S {};
  TemplateFn(S()); // compat-warning {{local type 'S' as template argument is incompatible with C++98}} \
                   // cxx03-warning {{template argument uses local type 'S'}} \
                   // expected-note {{while substituting deduced template arguments}}
}
struct {} obj_of_unnamed_type; // expected-note {{here}}
int UnnamedTemplateArg = TemplateFn(obj_of_unnamed_type); // compat-warning {{unnamed type as template argument is incompatible with C++98}} \
                                                          // cxx03-warning {{template argument uses unnamed type}} \
                                                          // expected-note {{while substituting deduced template arguments}}

// FIXME: We do not implement C++98 compatibility warnings for the C++17
// template argument evaluation rules.
namespace RedundantParensInAddressTemplateParam {
  int n;
  template<int*p> struct S {};
  S<(&n)> s; // pedantic-until-cxx14-warning {{parentheses around address non-type template argument are incompatible with C++98}}
  S<(((&n)))> t; // pedantic-until-cxx14-warning {{parentheses around address non-type template argument are incompatible with C++98}}
}

namespace TemplateSpecOutOfScopeNs {
  template<typename T> struct S {};
}
template<> struct TemplateSpecOutOfScopeNs::S<char> {};

struct Typename {
  template<typename T> struct Inner {};
};
typename ::Typename TypenameOutsideTemplate(); // pedantic-warning {{'typename' outside of a template is incompatible with C++98}}
Typename::template Inner<int> TemplateOutsideTemplate(); // pedantic-warning {{use of 'template' keyword outside of a template is incompatible with C++98}}

struct TrivialButNonPOD {
  int f(int);
private:
  int k;
};
void Ellipsis(int n, ...);
void TrivialButNonPODThroughEllipsis() {
  Ellipsis(1, TrivialButNonPOD()); // compat-warning {{passing object of trivial but non-POD type 'TrivialButNonPOD' through variadic function is incompatible with C++98}} \
                                   // cxx03-error {{cannot pass object of non-POD type 'TrivialButNonPOD' through variadic function; call will abort at runtime}}
}

struct HasExplicitConversion {
  explicit operator bool(); // pedantic-warning {{explicit conversion functions are incompatible with C++98}}
};

struct Struct {};
enum Enum {
  enum_val = 0, // pedantic-warning {{commas at the end of enumerator lists are incompatible with C++98}}
};
struct BadFriends {
  friend enum ::Enum; // compat-warning {{elaborated enum specifier cannot be declared as a friend}} \
                      // cxx03-warning {{elaborated enum specifier cannot be declared as a friend}} \
                      // expected-note {{remove 'enum' to befriend an enum}}
  friend int; // pedantic-warning {{non-class friend type 'int' is incompatible with C++98}}
  friend Struct; // compat-warning {{befriending 'Struct' without 'struct' keyword is incompatible with C++98}} \
                 // cxx03-warning {{unelaborated friend declaration is a C++11 extension; specify 'struct' to befriend 'Struct'}}
};

int n = {}; // compat-warning {{scalar initialized from empty initializer list is incompatible with C++98}} \
            // cxx03-error {{scalar initializer cannot be empty}}

class PrivateMember {
  struct ImPrivate {}; // cxx03-note {{implicitly declared private here}}
};
template<typename T> typename T::ImPrivate SFINAEAccessControl(T t) { // compat-warning {{substitution failure due to access control is incompatible with C++98}} \
                                                                      // cxx03-error {{'ImPrivate' is a private member of 'PrivateMember'}}
  return typename T::ImPrivate(); // cxx03-error {{'ImPrivate' is a private member of 'PrivateMember'}}
}
int SFINAEAccessControl(...) { return 0; }
int CheckSFINAEAccessControl = SFINAEAccessControl(PrivateMember()); // expected-note {{while substituting deduced template arguments into function template 'SFINAEAccessControl' [with T = PrivateMember]}} \
                                                                     // cxx03-error {{no viable conversion}} cxx03-note {{in instantiation}}

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
    NonTrivCtor ntc;  // compat-warning {{union member 'ntc' with a non-trivial default constructor is incompatible with C++98}} \
                      // cxx03-error {{non-trivial default constructor}}
    NonTrivCopy ntcp; // compat-warning {{union member 'ntcp' with a non-trivial copy constructor is incompatible with C++98}} \
                      // cxx03-error {{non-trivial copy constructor}}
    NonTrivDtor ntd;  // compat-warning {{union member 'ntd' with a non-trivial destructor is incompatible with C++98}} \
                      // cxx03-error {{non-trivial destructor}}
  };
  struct Wrap {
    struct {
      NonTrivCtor ntc;  // compat-warning {{anonymous struct member 'ntc' with a non-trivial default constructor is incompatible with C++98}} \
                        // cxx03-error {{non-trivial default constructor}}
      NonTrivCopy ntcp; // compat-warning {{anonymous struct member 'ntcp' with a non-trivial copy constructor is incompatible with C++98}} \
                        // cxx03-error {{non-trivial copy constructor}}
      NonTrivDtor ntd;  // compat-warning {{anonymous struct member 'ntd' with a non-trivial destructor is incompatible with C++98}} \
                        // cxx03-error {{non-trivial destructor}}
    };
  };
  union WithStaticDataMember {
    static constexpr double d = 0.0; // pedantic-warning {{static data member 'd' in union is incompatible with C++98}} \
                                     // compat-warning {{'constexpr' specifier is incompatible with C++98}} \
                                     // cxx03-error {{unknown type name 'constexpr'}}
    static const int n = 0; // pedantic-warning {{static data member 'n' in union is incompatible with C++98}}
    static int k; // pedantic-warning {{static data member 'k' in union is incompatible with C++98}}
  };
}

int EnumNNS = Enum::enum_val; // compat-warning {{enumeration type in nested name specifier is incompatible with C++98}} \
                              // cxx03-warning {{use of enumeration in a nested name specifier is a C++11 extension}}
template<typename T> void EnumNNSFn() {
  int k = T::enum_val; // compat-warning {{enumeration type in nested name specifier is incompatible with C++98}} \
                       // cxx03-error {{type 'Enum' cannot be used prior to '::' because it has no members}}
}
template void EnumNNSFn<Enum>(); // expected-note {{in instantiation}}

void JumpDiagnostics(int n) {
  goto DirectJump; // compat-warning {{jump from this goto statement to its label is incompatible with C++98}} \
                   // cxx03-error {{cannot jump from this goto statement to its label}}
  TrivialButNonPOD tnp1; // expected-note {{jump bypasses initialization of non-POD variable}}

DirectJump:
  void *Table[] = {&&DirectJump, &&Later};
  goto *Table[n]; // compat-warning {{jump from this indirect goto statement to one of its possible targets is incompatible with C++98}} \
                  // cxx03-error {{cannot jump from this indirect goto statement to one of its possible targets}}

  asm goto (""::::Later); // compat-warning {{jump from this asm goto statement to one of its possible targets is incompatible with C++98}} \
                          // cxx03-error {{cannot jump from this asm goto statement to one of its possible targets}}

  TrivialButNonPOD tnp2; // expected-note 2 {{jump bypasses initialization of non-POD variable}}
Later: // expected-note {{possible target of indirect goto statement}} \
       // expected-note {{possible target of asm goto statement}}
  switch (n) {
    TrivialButNonPOD tnp3; // expected-note {{jump bypasses initialization of non-POD variable}}
  default: // compat-warning {{jump from switch statement to this case label is incompatible with C++98}} \
           // cxx03-error {{cannot jump from switch statement to this case label}}
    return;
  }
}

namespace UnevaluatedMemberAccess {
  struct S {
    int n;
    int f() { return sizeof(S::n); } // ok
  };
  int k = sizeof(S::n); // compat-warning {{use of non-static data member 'n' in an unevaluated context is incompatible with C++98}} \
                        // cxx03-error {{invalid use of non-static data member 'n'}}
  const std::type_info &ti = typeid(S::n); // compat-warning {{use of non-static data member 'n' in an unevaluated context is incompatible with C++98}} \
                                           // cxx03-error {{invalid use of non-static data member 'n'}}
}

namespace LiteralUCNs {
  char c1 = '\u001e'; // compat-warning {{universal character name referring to a control character is incompatible with C++98}} \
                      // cxx03-error {{universal character name refers to a control character}}
  wchar_t c2 = L'\u0041'; // compat-warning {{specifying character 'A' with a universal character name is incompatible with C++98}} \
                          // cxx03-error {{character 'A' cannot be specified by a universal character name}}
  const char *s1 = "foo\u0031"; // compat-warning {{specifying character '1' with a universal character name is incompatible with C++98}} \
                                // cxx03-error {{character '1' cannot be specified by a universal character name}}
  const wchar_t *s2 = L"bar\u0085"; // compat-warning {{universal character name referring to a control character is incompatible with C++98}} \
                                    // cxx03-error {{universal character name refers to a control character}}
}

// FIXME: We do not implement C++98 compatibility warnings for the C++17
// template argument evaluation rules.
namespace NonTypeTemplateArgs {
  template<typename T, T v> struct S {};
  const int k = 5;     // pedantic-until-cxx14-note {{here}}
  static void f() {}   // pedantic-until-cxx14-note {{here}}
  S<const int&, k> s1; // pedantic-until-cxx14-warning {{non-type template argument referring to object 'k' with internal linkage is incompatible with C++98}}
  S<void(&)(), f> s2;  // pedantic-until-cxx14-warning {{non-type template argument referring to function 'f' with internal linkage is incompatible with C++98}}
}

namespace NullPointerTemplateArg {
  struct A {};
  template<int*> struct X {}; // cxx03-note {{template parameter is declared here}}
  template<int A::*> struct Y {};
  X<(int*)0> x; // until-cxx14-warning {{use of null pointer as non-type template argument is incompatible with C++98}} \
                // cxx03-error {{non-type template argument does not refer to any declaration}}
  Y<(int A::*)0> y; // until-cxx14-warning {{use of null pointer as non-type template argument is incompatible with C++98}} \
                    // cxx03-error {{non-type template argument is not a pointer to member constant}}
}

namespace PR13480 {
  struct basic_iterator {
    basic_iterator(const basic_iterator &it) {} // expected-note {{because type 'PR13480::basic_iterator' has a user-provided copy constructor}}
    basic_iterator(basic_iterator &it) {}
  };

  union test {
    basic_iterator it; // compat-warning {{union member 'it' with a non-trivial copy constructor is incompatible with C++98}} \
                       // cxx03-error {{union member 'it' has a non-trivial copy constructor}}
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
    a x; // compat-warning {{union member 'x' with a non-trivial copy assignment operator is incompatible with C++98}} \
         // cxx03-error {{union member 'x' has a non-trivial copy assignment operator}}
    b y; // compat-warning {{union member 'y' with a non-trivial copy assignment operator is incompatible with C++98}} \
         // cxx03-error {{union member 'y' has a non-trivial copy assignment operator}}
  };
}

namespace rdar11736429 {
  struct X { // expected-note {{because type 'rdar11736429::X' has no default constructor}}
    X(const X&) = delete; // pedantic-warning {{deleted function definitions are incompatible with C++98}} \
                          // expected-note {{implicit default constructor suppressed by user-declared constructor}}
  };

  union S {
    X x; // compat-warning {{union member 'x' with a non-trivial default constructor is incompatible with C++98}} \
         // cxx03-error {{union member 'x' has a non-trivial default constructor}}
  };
}

#define VA_MACRO(x, ...) x // pedantic-warning {{variadic macros are incompatible with C++98}}
VA_MACRO(,x) // pedantic-warning {{empty macro arguments are incompatible with C++98}}

; // pedantic-warning {{extra ';' outside of a function is incompatible with C++98}}

enum EnumFixed : int { // pedantic-warning {{enumeration types with a fixed underlying type are incompatible with C++98}}
};

void *dlsym();
void (*FnPtr)() = (void(*)())dlsym(); // pedantic-warning {{cast between pointer-to-function and pointer-to-object is incompatible with C++98}}
void *FnVoidPtr = (void*)&dlsym; // pedantic-warning {{cast between pointer-to-function and pointer-to-object is incompatible with C++98}}

struct ConvertToInt {
  operator int();
};
int *ArraySizeConversion = new int[ConvertToInt()];
// pedantic-warning-re@-1 {{implicit conversion from array size expression of type 'ConvertToInt' to integral type '{{(int|size_t)}}' is incompatible with C++98}}

template<typename T> class ExternTemplate {};
extern template class ExternTemplate<int>; // pedantic-warning {{extern templates are incompatible with C++98}}

long long ll1 = // pedantic-warning {{'long long' is incompatible with C++98}}
         -42LL; // pedantic-warning {{'long long' is incompatible with C++98}}
unsigned long long ull1 = // pedantic-warning {{'long long' is incompatible with C++98}}
                   42ULL; // pedantic-warning {{'long long' is incompatible with C++98}}

namespace CopyCtorIssues {
  struct Private {
    Private();
  private:
    Private(const Private&); // pedantic-note {{declared private here}}
  };
  struct NoViable {
    NoViable();          // pedantic-note {{not viable}}
    NoViable(NoViable&); // pedantic-note {{not viable}}
  };
  struct Ambiguous {
    Ambiguous();
    Ambiguous(const Ambiguous &, int = 0);    // pedantic-note {{candidate}} cxx03-note {{candidate}}
    Ambiguous(const Ambiguous &, double = 0); // pedantic-note {{candidate}} cxx03-note {{candidate}}
  };
  struct Deleted {
    Private p; // pedantic-note {{implicitly deleted}}
  };

  const Private &a = Private(); // pedantic-warning {{copying variable of type 'Private' when binding a reference to a temporary would invoke an inaccessible constructor in C++98}}
  const NoViable &b = NoViable(); // pedantic-warning {{copying variable of type 'NoViable' when binding a reference to a temporary would find no viable constructor in C++98}}
  const Ambiguous &c = Ambiguous(); // pedantic-warning {{copying variable of type 'Ambiguous' when binding a reference to a temporary would find ambiguous constructors in C++98}} \
                                    // cxx03-error {{ambiguous constructor call when copying variable of type 'Ambiguous'}}
  const Deleted &d = Deleted(); // pedantic-warning {{copying variable of type 'Deleted' when binding a reference to a temporary would invoke a deleted constructor in C++98}}
}

int with_attribute [[ ]]; // pedantic-warning {{[[]] attributes are incompatible with C++98}}

#line 32767 // ok
#line 32768 // pedantic-warning {{#line number greater than 32767 is incompatible with C++98}}
