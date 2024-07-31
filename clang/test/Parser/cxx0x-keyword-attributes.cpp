// RUN: sed -e "s@ATTR_USE@__arm_streaming@g" -e "s@ATTR_NAME@__arm_streaming@g" %s > %t
// RUN: %clang_cc1 -fcxx-exceptions -fdeclspec -fexceptions -fsyntax-only -verify -std=c++11 -Wc++14-compat -Wc++14-extensions -Wc++17-extensions -triple aarch64-none-linux-gnu -target-feature +sme -x c++ %t
// RUN: sed -e "s@ATTR_USE@__arm_inout\(\"za\"\)@g" -e "s@ATTR_NAME@__arm_inout@g" %s > %t
// RUN: %clang_cc1 -fcxx-exceptions -fdeclspec -fexceptions -fsyntax-only -verify -std=c++11 -Wc++14-compat -Wc++14-extensions -Wc++17-extensions -triple aarch64-none-linux-gnu -target-feature +sme -x c++ %t

// Need std::initializer_list
namespace std {
  typedef decltype(sizeof(int)) size_t;

  // libc++'s implementation
  template <class _E>
  class initializer_list
  {
    const _E* __begin_;
    size_t    __size_;

    initializer_list(const _E* __b, size_t __s)
      : __begin_(__b),
        __size_(__s)
    {}

  public:
    typedef _E        value_type;
    typedef const _E& reference;
    typedef const _E& const_reference;
    typedef size_t    size_type;

    typedef const _E* iterator;
    typedef const _E* const_iterator;

    initializer_list() : __begin_(nullptr), __size_(0) {}

    size_t    size()  const {return __size_;}
    const _E* begin() const {return __begin_;}
    const _E* end()   const {return __begin_ + __size_;}
  };
}


// Declaration syntax checks
ATTR_USE int before_attr; // expected-error {{'ATTR_NAME' only applies to function types}}
int ATTR_USE between_attr; // expected-error {{'ATTR_NAME' only applies to function types}}
const ATTR_USE int between_attr_2 = 0; // expected-error {{'ATTR_NAME' cannot appear here}}
int after_attr ATTR_USE; // expected-error {{'ATTR_NAME' only applies to function types}}
int * ATTR_USE ptr_attr; // expected-error {{'ATTR_NAME' only applies to function types}}
int & ATTR_USE ref_attr = after_attr; // expected-error {{'ATTR_NAME' only applies to function types}}
int && ATTR_USE rref_attr = 0; // expected-error {{'ATTR_NAME' only applies to function types}}
int array_attr [1] ATTR_USE; // expected-error {{'ATTR_NAME' only applies to function types}}
void fn_attr () ATTR_USE;
void noexcept_fn_attr () noexcept ATTR_USE;
struct MemberFnOrder {
  virtual void f() const volatile && noexcept ATTR_USE final = 0;
};
struct ATTR_USE struct_attr; // expected-error {{'ATTR_NAME' only applies to non-K&R-style functions}}
class ATTR_USE class_attr {}; // expected-error {{'ATTR_NAME' only applies to non-K&R-style functions}}
union ATTR_USE union_attr; // expected-error {{'ATTR_NAME' only applies to non-K&R-style functions}}
enum ATTR_USE E { }; // expected-error {{'ATTR_NAME' only applies to non-K&R-style functions}}
namespace test_misplacement {
ATTR_USE struct struct_attr2;  // expected-error {{misplaced 'ATTR_NAME'}}
ATTR_USE class class_attr2; // expected-error {{misplaced 'ATTR_NAME'}}
ATTR_USE union union_attr2; // expected-error {{misplaced 'ATTR_NAME'}}
ATTR_USE enum  E2 { }; // expected-error {{misplaced 'ATTR_NAME'}}
}

// Checks attributes placed at wrong syntactic locations of class specifiers.
class ATTR_USE ATTR_USE // expected-error 2 {{'ATTR_NAME' only applies to non-K&R-style functions}}
  attr_after_class_name_decl ATTR_USE ATTR_USE; // expected-error {{'ATTR_NAME' cannot appear here}} \
                                                                 expected-error 2 {{'ATTR_NAME' only applies to non-K&R-style functions}}

class ATTR_USE ATTR_USE // expected-error 2 {{'ATTR_NAME' only applies to non-K&R-style functions}}
 attr_after_class_name_definition ATTR_USE ATTR_USE ATTR_USE{}; // expected-error {{'ATTR_NAME' cannot appear here}} \
                                                                                        expected-error 3 {{'ATTR_NAME' only applies to non-K&R-style functions}}

class ATTR_USE c {}; // expected-error {{'ATTR_NAME' only applies to non-K&R-style functions}}
class c ATTR_USE ATTR_USE x; // expected-error 2 {{'ATTR_NAME' only applies to function types}}
class c ATTR_USE ATTR_USE y ATTR_USE ATTR_USE; // expected-error 4 {{'ATTR_NAME' only applies to function types}}
class c final [(int){0}];

class base {};
class ATTR_USE ATTR_USE final_class // expected-error 2 {{'ATTR_NAME' only applies to non-K&R-style functions}}
  ATTR_USE alignas(float) final // expected-error {{'ATTR_NAME' cannot appear here}} \
                                          expected-error {{'ATTR_NAME' only applies to non-K&R-style functions}}
  ATTR_USE alignas(float) ATTR_USE alignas(float): base{}; // expected-error {{'ATTR_NAME' cannot appear here}}

class ATTR_USE ATTR_USE final_class_another // expected-error 2 {{'ATTR_NAME' only applies to non-K&R-style functions}}
  ATTR_USE ATTR_USE alignas(16) final // expected-error {{'ATTR_NAME' cannot appear here}} \
                                                       expected-error 2 {{'ATTR_NAME' only applies to non-K&R-style functions}}
  ATTR_USE ATTR_USE alignas(16) ATTR_USE{}; // expected-error {{'ATTR_NAME' cannot appear here}}

class after_class_close {} ATTR_USE; // expected-error {{'ATTR_NAME' cannot appear here, place it after "class" to apply it to the type declaration}}

class C {};

ATTR_USE struct with_init_declarators {} init_declarator; // expected-error {{'ATTR_NAME' only applies to function types}}
ATTR_USE struct no_init_declarators; // expected-error {{misplaced 'ATTR_NAME'}}
template<typename> ATTR_USE struct no_init_declarators_template; // expected-error {{'ATTR_NAME' cannot appear here}}
void fn_with_structs() {
  ATTR_USE struct with_init_declarators {} init_declarator; // expected-error {{'ATTR_NAME' only applies to function types}}
  ATTR_USE struct no_init_declarators; // expected-error {{'ATTR_NAME' cannot appear here}}
}
ATTR_USE; // expected-error {{'ATTR_NAME' only applies to non-K&R-style functions}}
struct ctordtor {
  ATTR_USE ctordtor ATTR_USE () ATTR_USE; // expected-error 2 {{'ATTR_NAME' cannot be applied to a declaration}}
  ctordtor (C) ATTR_USE;
  ATTR_USE ~ctordtor ATTR_USE () ATTR_USE; // expected-error 2 {{'ATTR_NAME' cannot be applied to a declaration}}
};
ATTR_USE ctordtor::ctordtor ATTR_USE () ATTR_USE {} // expected-error 2 {{'ATTR_NAME' cannot be applied to a declaration}}
ATTR_USE ctordtor::ctordtor (C) ATTR_USE try {} catch (...) {} // expected-error {{'ATTR_NAME' cannot be applied to a declaration}}
ATTR_USE ctordtor::~ctordtor ATTR_USE () ATTR_USE {} // expected-error 2 {{'ATTR_NAME' cannot be applied to a declaration}}
extern "C++" ATTR_USE int extern_attr; // expected-error {{'ATTR_NAME' only applies to function types}}
template <typename T> ATTR_USE void template_attr (); // expected-error {{'ATTR_NAME' cannot be applied to a declaration}}
ATTR_USE ATTR_USE int ATTR_USE ATTR_USE multi_attr ATTR_USE ATTR_USE; // expected-error 6 {{'ATTR_NAME' only applies to function types}}

int (paren_attr) ATTR_USE; // expected-error {{'ATTR_NAME' cannot appear here}}
unsigned ATTR_USE int attr_in_decl_spec; // expected-error {{'ATTR_NAME' cannot appear here}}
unsigned ATTR_USE int ATTR_USE const double_decl_spec = 0; // expected-error 2 {{'ATTR_NAME' cannot appear here}}
class foo {
  void const_after_attr () ATTR_USE const; // expected-error {{expected ';'}}
};
extern "C++" ATTR_USE { } // expected-error {{'ATTR_NAME' cannot appear here}}
ATTR_USE extern "C++" { } // expected-error {{'ATTR_NAME' cannot appear here}}
ATTR_USE template <typename T> void before_template_attr (); // expected-error {{'ATTR_NAME' cannot appear here}}
ATTR_USE namespace ns { int i; } // expected-error {{'ATTR_NAME' cannot appear here}}
ATTR_USE static_assert(true, ""); //expected-error {{'ATTR_NAME' cannot appear here}}
ATTR_USE asm(""); // expected-error {{'ATTR_NAME' cannot appear here}}

ATTR_USE using ns::i; // expected-warning {{ISO C++}} \
                                expected-error {{'ATTR_NAME' only applies to non-K&R-style functions}}
ATTR_USE using namespace ns; // expected-error {{'ATTR_NAME' only applies to non-K&R-style functions}}
namespace ATTR_USE ns2 {} // expected-warning {{attributes on a namespace declaration are a C++17 extension}} \
                                    expected-error {{'ATTR_NAME' only applies to non-K&R-style functions}}

using ATTR_USE alignas(4)ATTR_USE ns::i;          // expected-warning 2 {{ISO C++}} \
                                                                   expected-error {{'ATTR_NAME' cannot appear here}} \
                                                                   expected-error {{'alignas' attribute only applies to variables, data members and tag types}} \
                                                                   expected-warning {{ISO C++}} \
                                                                   expected-error 2 {{'ATTR_NAME' only applies to non-K&R-style functions}}
using ATTR_USE alignas(4) ATTR_USE foobar = int; // expected-error {{'ATTR_NAME' cannot appear here}} \
                                                                  expected-error {{'alignas' attribute only applies to}} \
                                                                  expected-error 2 {{'ATTR_NAME' only applies to function types}}

ATTR_USE using T = int; // expected-error {{'ATTR_NAME' cannot appear here}}
using T ATTR_USE = int; // expected-error {{'ATTR_NAME' only applies to function types}}
template<typename T> using U ATTR_USE = T; // expected-error {{'ATTR_NAME' only applies to function types}}
using ns::i ATTR_USE; // expected-warning {{ISO C++}} \
                                expected-error {{'ATTR_NAME' only applies to non-K&R-style functions}}
using ns::i ATTR_USE, ns::i ATTR_USE; // expected-warning 2 {{ISO C++}} \
                                                       expected-warning {{use of multiple declarators in a single using declaration is a C++17 extension}} \
                                                       expected-error 2 {{'ATTR_NAME' only applies to non-K&R-style functions}}
struct using_in_struct_base {
  typedef int i, j, k, l;
};
struct using_in_struct : using_in_struct_base {
  ATTR_USE using using_in_struct_base::i; // expected-warning {{ISO C++}} \
                                                    expected-error {{'ATTR_NAME' only applies to non-K&R-style functions}}
  using using_in_struct_base::j ATTR_USE; // expected-warning {{ISO C++}} \
                                                    expected-error {{'ATTR_NAME' only applies to non-K&R-style functions}}
  ATTR_USE using using_in_struct_base::k ATTR_USE, using_in_struct_base::l ATTR_USE; // expected-warning 3 {{ISO C++}} \
                                                                                                             expected-warning {{use of multiple declarators in a single using declaration is a C++17 extension}} \
                                                                                                             expected-error 4 {{'ATTR_NAME' only applies to non-K&R-style functions}}
};
using ATTR_USE ns::i; // expected-warning {{ISO C++}} \
                                expected-error {{'ATTR_NAME' cannot appear here}} \
                                expected-error {{'ATTR_NAME' only applies to non-K&R-style functions}}
using T ATTR_USE = int; // expected-error {{'ATTR_NAME' only applies to function types}}

auto trailing() -> ATTR_USE const int; // expected-error {{'ATTR_NAME' cannot appear here}}
auto trailing() -> const ATTR_USE int; // expected-error {{'ATTR_NAME' cannot appear here}}
auto trailing() -> const int ATTR_USE; // expected-error {{'ATTR_NAME' only applies to function types}}
auto trailing_2() -> struct struct_attr ATTR_USE; // expected-error {{'ATTR_NAME' only applies to function types}}

namespace N {
  struct S {};
};
template<typename> struct Template {};

// FIXME: Improve this diagnostic
struct ATTR_USE N::S s; // expected-error {{'ATTR_NAME' cannot appear here}}
struct ATTR_USE Template<int> t; // expected-error {{'ATTR_NAME' cannot appear here}}
struct ATTR_USE ::template Template<int> u; // expected-error {{'ATTR_NAME' cannot appear here}}
template struct ATTR_USE Template<char>; // expected-error {{'ATTR_NAME' cannot appear here}}
template struct __attribute__((pure)) Template<std::size_t>; // We still allow GNU-style attributes here
template <> struct ATTR_USE Template<void>; // expected-error {{'ATTR_NAME' only applies to non-K&R-style functions}}

enum ATTR_USE E1 {}; // expected-error {{'ATTR_NAME' only applies to non-K&R-style functions}}
enum ATTR_USE E2; // expected-error {{forbids forward references}} \
                            expected-error {{'ATTR_NAME' only applies to non-K&R-style functions}}
enum ATTR_USE E1; // expected-error {{'ATTR_NAME' only applies to non-K&R-style functions}}
enum ATTR_USE E3 : int; // expected-error {{'ATTR_NAME' only applies to non-K&R-style functions}}
enum ATTR_USE { // expected-error {{'ATTR_NAME' only applies to non-K&R-style functions}}
  k_123 ATTR_USE = 123 // expected-warning {{attributes on an enumerator declaration are a C++17 extension}} \
                                 expected-error {{'ATTR_NAME' only applies to non-K&R-style functions}}
};
enum ATTR_USE E1 e; // expected-error {{'ATTR_NAME' cannot appear here}}
enum ATTR_USE class E4 { }; // expected-error {{'ATTR_NAME' cannot appear here}}
enum struct ATTR_USE E5; // expected-error {{'ATTR_NAME' only applies to non-K&R-style functions}}
enum E6 {} ATTR_USE; // expected-error {{'ATTR_NAME' cannot appear here, place it after "enum" to apply it to the type declaration}}

struct S {
  friend int f ATTR_USE (); // expected-error {{'ATTR_NAME' cannot appear here}} \
                                      expected-error {{'ATTR_NAME' cannot be applied to a declaration}}
  friend int f2 ATTR_USE () {} // expected-error {{'ATTR_NAME' cannot be applied to a declaration}}
  ATTR_USE friend int g(); // expected-error {{'ATTR_NAME' cannot appear here}}
  ATTR_USE friend int h() { // expected-error {{'ATTR_NAME' cannot be applied to a declaration}}
  }
  ATTR_USE friend int f3(), f4(), f5(); // expected-error {{'ATTR_NAME' cannot appear here}}
  friend int f6 ATTR_USE (), f7 ATTR_USE (), f8 ATTR_USE (); // expected-error3 {{'ATTR_NAME' cannot appear here}} \
                                                                                     expected-error 3 {{'ATTR_NAME' cannot be applied to a declaration}}
  friend class ATTR_USE C; // expected-error {{'ATTR_NAME' cannot appear here}}
  ATTR_USE friend class D; // expected-error {{'ATTR_NAME' cannot appear here}}
  ATTR_USE friend int; // expected-error {{'ATTR_NAME' cannot appear here}}
};
template<typename T> void tmpl (T) {}
template ATTR_USE void tmpl(char); // expected-error {{'ATTR_NAME' cannot appear here}}
template void ATTR_USE tmpl(short); // expected-error {{'ATTR_NAME' only applies to function types}}

// Statement tests
void foo () {
  ATTR_USE ; // expected-error {{'ATTR_NAME' cannot be applied to a statement}}
  ATTR_USE { } // expected-error {{'ATTR_NAME' cannot be applied to a statement}}
  ATTR_USE if (0) { } // expected-error {{'ATTR_NAME' cannot be applied to a statement}}
  ATTR_USE for (;;); // expected-error {{'ATTR_NAME' cannot be applied to a statement}}
  ATTR_USE do { // expected-error {{'ATTR_NAME' cannot be applied to a statement}}
    ATTR_USE continue; // expected-error {{'ATTR_NAME' cannot be applied to a statement}}
  } while (0);
  ATTR_USE while (0); // expected-error {{'ATTR_NAME' cannot be applied to a statement}}

  ATTR_USE switch (i) { // expected-error {{'ATTR_NAME' cannot be applied to a statement}}
    ATTR_USE case 0: // expected-error {{'ATTR_NAME' cannot be applied to a statement}}
    ATTR_USE default: // expected-error {{'ATTR_NAME' cannot be applied to a statement}}
      ATTR_USE break; // expected-error {{'ATTR_NAME' cannot be applied to a statement}}
  }

  ATTR_USE goto there; // expected-error {{'ATTR_NAME' cannot be applied to a statement}}
  ATTR_USE there: // expected-error {{'ATTR_NAME' only applies to non-K&R-style functions}}

  ATTR_USE try { // expected-error {{'ATTR_NAME' cannot be applied to a statement}}
  } ATTR_USE catch (...) { // expected-error {{'ATTR_NAME' cannot appear here}}
  }

  void bar ATTR_USE (ATTR_USE int i, ATTR_USE int j); // expected-error 2 {{'ATTR_NAME' only applies to function types}} \
                                                                              expected-error {{'ATTR_NAME' cannot be applied to a declaration}}
  using FuncType = void (ATTR_USE int); // expected-error {{'ATTR_NAME' only applies to function types}}
  void baz(ATTR_USE...); // expected-error {{expected parameter declarator}}

  ATTR_USE return; // expected-error {{'ATTR_NAME' cannot be applied to a statement}}
}

// Expression tests
void bar () {
  new int[42]ATTR_USE[5]ATTR_USE{}; // expected-error {{'ATTR_NAME' only applies to function types}}
}

// Condition tests
void baz () {
  if (ATTR_USE bool b = true) { // expected-error {{'ATTR_NAME' only applies to function types}}
    switch (ATTR_USE int n { 42 }) { // expected-error {{'ATTR_NAME' only applies to function types}}
    default:
      for (ATTR_USE int n = 0; ATTR_USE char b = n < 5; ++b) { // expected-error 2 {{'ATTR_NAME' only applies to function types}}
      }
    }
  }
  int x;
  // An attribute can be applied to an expression-statement, such as the first
  // statement in a for. But it can't be applied to a condition which is an
  // expression.
  for (ATTR_USE x = 0; ; ) {} // expected-error {{'ATTR_NAME' cannot appear here}}
  for (; ATTR_USE x < 5; ) {} // expected-error {{'ATTR_NAME' cannot appear here}}
  while (ATTR_USE bool k { false }) { // expected-error {{'ATTR_NAME' only applies to function types}}
  }
  while (ATTR_USE true) { // expected-error {{'ATTR_NAME' cannot appear here}}
  }
  do {
  } while (ATTR_USE false); // expected-error {{'ATTR_NAME' cannot appear here}}

  for (ATTR_USE int n : { 1, 2, 3 }) { // expected-error {{'ATTR_NAME' only applies to function types}}
  }
}

enum class __attribute__((visibility("hidden"))) SecretKeepers {
  one, /* rest are deprecated */ two, three
};
enum class ATTR_USE EvenMoreSecrets {}; // expected-error {{'ATTR_NAME' only applies to non-K&R-style functions}}

// Forbid attributes on decl specifiers.
unsigned ATTR_USE static int ATTR_USE v1; // expected-error {{'ATTR_NAME' only applies to function types}} \
           expected-error {{'ATTR_NAME' cannot appear here}}
typedef ATTR_USE unsigned long ATTR_USE v2; // expected-error {{'ATTR_NAME' only applies to function types}} \
          expected-error {{'ATTR_NAME' cannot appear here}}
int ATTR_USE foo(int ATTR_USE x); // expected-error 2 {{'ATTR_NAME' only applies to function types}}

ATTR_USE; // expected-error {{'ATTR_NAME' only applies to non-K&R-style functions}}

class A {
  A(ATTR_USE int a); // expected-error {{'ATTR_NAME' only applies to function types}}
};
A::A(ATTR_USE int a) {} // expected-error {{'ATTR_NAME' only applies to function types}}

template<typename T> struct TemplateStruct {};
class FriendClassesWithAttributes {
  // We allow GNU-style attributes here
  template <class _Tp, class _Alloc> friend class __attribute__((__type_visibility__("default"))) vector;
  template <class _Tp, class _Alloc> friend class __declspec(code_seg("foo,whatever")) vector2;
  // But not C++11 ones
  template <class _Tp, class _Alloc> friend class ATTR_USE vector3;                                         // expected-error {{'ATTR_NAME' cannot appear here}}

  // Also allowed
  friend struct __attribute__((__type_visibility__("default"))) TemplateStruct<FriendClassesWithAttributes>;
  friend struct __declspec(code_seg("foo,whatever")) TemplateStruct<FriendClassesWithAttributes>;
  friend struct ATTR_USE TemplateStruct<FriendClassesWithAttributes>;                                       // expected-error {{'ATTR_NAME' cannot appear here}}
};

// Check ordering: C++11 attributes must appear before GNU attributes.
class Ordering {
  void f1(
    int (ATTR_USE __attribute__(()) int n) // expected-error {{'ATTR_NAME' only applies to function types}}
  ) {
  }

  void f2(
      int (*)(ATTR_USE __attribute__(()) int n) // expected-error {{'ATTR_NAME' only applies to function types}}
  ) {
  }

  void f3(
    int (__attribute__(()) ATTR_USE int n) // expected-error {{'ATTR_NAME' cannot appear here}}
  ) {
  }

  void f4(
      int (*)(__attribute__(()) ATTR_USE int n) // expected-error {{'ATTR_NAME' cannot appear here}}
  ) {
  }
};

namespace base_specs {
struct A {};
struct B : ATTR_USE A {}; // expected-error {{'ATTR_NAME' cannot be applied to a base specifier}}
struct C : ATTR_USE virtual A {}; // expected-error {{'ATTR_NAME' cannot be applied to a base specifier}}
struct D : ATTR_USE public virtual A {}; // expected-error {{'ATTR_NAME' cannot be applied to a base specifier}}
struct E : public ATTR_USE virtual A {}; // expected-error {{'ATTR_NAME' cannot appear here}} \
                                                   expected-error {{'ATTR_NAME' cannot be applied to a base specifier}}
struct F : virtual ATTR_USE public A {}; // expected-error {{'ATTR_NAME' cannot appear here}} \
                                                   expected-error {{'ATTR_NAME' cannot be applied to a base specifier}}
}

namespace ATTR_USE ns_attr {}; // expected-error {{'ATTR_NAME' only applies to non-K&R-style functions}} \
                                         expected-warning {{attributes on a namespace declaration are a C++17 extension}}
