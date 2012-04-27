// RUN: %clang_cc1 -fsyntax-only -std=c++11 -Wc++98-compat -verify %s
// RUN: %clang_cc1 -fsyntax-only -std=c++11 %s

namespace std { struct type_info; }

template<typename ...T>  // expected-warning {{variadic templates are incompatible with C++98}}
class Variadic1 {};

template<template<typename> class ...T>  // expected-warning {{variadic templates are incompatible with C++98}}
class Variadic2 {};

template<int ...I>  // expected-warning {{variadic templates are incompatible with C++98}}
class Variadic3 {};

int alignas(8) with_alignas; // expected-warning {{'alignas' is incompatible with C++98}}
int with_attribute [[ ]]; // expected-warning {{attributes are incompatible with C++98}}

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
}

int InitList(int i = {}) { // expected-warning {{generalized initializer lists are incompatible with C++98}} \
                           // expected-warning {{scalar initialized from empty initializer list is incompatible with C++98}}
  (void)new int {}; // expected-warning {{generalized initializer lists are incompatible with C++98}} \
                    // expected-warning {{scalar initialized from empty initializer list is incompatible with C++98}}
  (void)int{}; // expected-warning {{generalized initializer lists are incompatible with C++98}} \
               // expected-warning {{scalar initialized from empty initializer list is incompatible with C++98}}
  int x { 0 }; // expected-warning {{generalized initializer lists are incompatible with C++98}}
  S<int> s = {}; // ok, aggregate
  s = {}; // expected-warning {{generalized initializer lists are incompatible with C++98}}
  return { 0 }; // expected-warning {{generalized initializer lists are incompatible with C++98}}
}
struct DelayedDefaultArgumentParseInitList {
  void f(int i = {1}) { // expected-warning {{generalized initializer lists are incompatible with C++98}}
  }
};

int operator"" _hello(const char *); // expected-warning {{literal operators are incompatible with C++98}}

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

void RangeFor() {
  int xs[] = {1, 2, 3};
  for (int &a : xs) { // expected-warning {{range-based for loop is incompatible with C++98}}
  }
}

struct InClassInit {
  int n = 0; // expected-warning {{in-class initialization of non-static data members is incompatible with C++98}}
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
static_assert(true, "!"); // expected-warning {{static_assert declarations are incompatible with C++98}}

// FIXME: Reintroduce this test if support for inheriting constructors is
//        implemented.
#if 0
struct InhCtorBase {
  InhCtorBase(int);
};
struct InhCtorDerived : InhCtorBase {
  using InhCtorBase::InhCtorBase; // xpected-warning {{inheriting constructors are incompatible with C++98}}
};
#endif

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

namespace RedundantParensInAddressTemplateParam {
  int n;
  template<int*p> struct S {};
  S<(&n)> s; // expected-warning {{redundant parentheses surrounding address non-type template argument are incompatible with C++98}}
  S<(((&n)))> t; // expected-warning {{redundant parentheses surrounding address non-type template argument are incompatible with C++98}}
}

namespace TemplateSpecOutOfScopeNs {
  template<typename T> struct S {}; // expected-note {{here}}
}
template<> struct TemplateSpecOutOfScopeNs::S<char> {}; // expected-warning {{class template specialization of 'S' outside namespace 'TemplateSpecOutOfScopeNs' is incompatible with C++98}}

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
  friend enum ::Enum; // expected-warning {{befriending enumeration type 'enum ::Enum' is incompatible with C++98}}
  friend int; // expected-warning {{non-class friend type 'int' is incompatible with C++98}}
  friend Struct; // expected-warning {{befriending 'Struct' without 'struct' keyword is incompatible with C++98}}
};

int n = {}; // expected-warning {{scalar initialized from empty initializer list is incompatible with C++98}}

class PrivateMember {
  struct ImPrivate {};
};
template<typename T> typename T::ImPrivate SFINAEAccessControl(T t) { // expected-warning {{substitution failure due to access control is incompatible with C++98}} expected-note {{while substituting deduced template arguments into function template 'SFINAEAccessControl' [with T = PrivateMember]}}
  return typename T::ImPrivate();
}
int SFINAEAccessControl(...) { return 0; }
int CheckSFINAEAccessControl = SFINAEAccessControl(PrivateMember());

template<typename T>
struct FriendRedefinition {
  friend void Friend() {} // expected-warning {{friend function 'Friend' would be implicitly redefined in C++98}} expected-note {{previous}}
};
FriendRedefinition<int> FriendRedef1;
FriendRedefinition<char> FriendRedef2; // expected-note {{requested here}}

namespace CopyCtorIssues {
  struct Private {
    Private();
  private:
    Private(const Private&); // expected-note {{declared private here}}
  };
  struct NoViable {
    NoViable();
    NoViable(NoViable&); // expected-note {{not viable}}
  };
  struct Ambiguous {
    Ambiguous();
    Ambiguous(const Ambiguous &, int = 0); // expected-note {{candidate}}
    Ambiguous(const Ambiguous &, double = 0); // expected-note {{candidate}}
  };
  struct Deleted {
    Private p; // expected-note {{implicitly deleted}}
  };

  const Private &a = Private(); // expected-warning {{copying variable of type 'CopyCtorIssues::Private' when binding a reference to a temporary would invoke an inaccessible constructor in C++98}}
  const NoViable &b = NoViable(); // expected-warning {{copying variable of type 'CopyCtorIssues::NoViable' when binding a reference to a temporary would find no viable constructor in C++98}}
  const Ambiguous &c = Ambiguous(); // expected-warning {{copying variable of type 'CopyCtorIssues::Ambiguous' when binding a reference to a temporary would find ambiguous constructors in C++98}}
  const Deleted &d = Deleted(); // expected-warning {{copying variable of type 'CopyCtorIssues::Deleted' when binding a reference to a temporary would invoke a deleted constructor in C++98}}
}

namespace UnionOrAnonStructMembers {
  struct NonTrivCtor {
    NonTrivCtor(); // expected-note 2{{user-declared constructor}}
  };
  struct NonTrivCopy {
    NonTrivCopy(const NonTrivCopy&); // expected-note 2{{user-declared copy constructor}}
  };
  struct NonTrivDtor {
    ~NonTrivDtor(); // expected-note 2{{user-declared destructor}}
  };
  union BadUnion {
    NonTrivCtor ntc; // expected-warning {{union member 'ntc' with a non-trivial constructor is incompatible with C++98}}
    NonTrivCopy ntcp; // expected-warning {{union member 'ntcp' with a non-trivial copy constructor is incompatible with C++98}}
    NonTrivDtor ntd; // expected-warning {{union member 'ntd' with a non-trivial destructor is incompatible with C++98}}
  };
  struct Wrap {
    struct {
      NonTrivCtor ntc; // expected-warning {{anonymous struct member 'ntc' with a non-trivial constructor is incompatible with C++98}}
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
  goto DirectJump; // expected-warning {{goto would jump into protected scope in C++98}}
  TrivialButNonPOD tnp1; // expected-note {{jump bypasses initialization of non-POD variable}}

DirectJump:
  void *Table[] = {&&DirectJump, &&Later};
  goto *Table[n]; // expected-warning {{indirect goto might cross protected scopes in C++98}}

  TrivialButNonPOD tnp2; // expected-note {{jump bypasses initialization of non-POD variable}}
Later: // expected-note {{possible target of indirect goto}}
  switch (n) {
    TrivialButNonPOD tnp3; // expected-note {{jump bypasses initialization of non-POD variable}}
  default: // expected-warning {{switch case would be in a protected scope in C++98}}
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

namespace NonTypeTemplateArgs {
  template<typename T, T v> struct S {};
  const int k = 5; // expected-note {{here}}
  static void f() {} // expected-note {{here}}
  S<const int&, k> s1; // expected-warning {{non-type template argument referring to object 'k' with internal linkage is incompatible with C++98}}
  S<void(&)(), f> s2; // expected-warning {{non-type template argument referring to function 'f' with internal linkage is incompatible with C++98}}
}
