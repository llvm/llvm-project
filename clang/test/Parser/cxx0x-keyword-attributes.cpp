// RUN: %clang_cc1 -fcxx-exceptions -fdeclspec -fexceptions -fsyntax-only -verify -std=c++11 -Wc++14-compat -Wc++14-extensions -Wc++17-extensions -triple aarch64-none-linux-gnu -target-feature +sme %s

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
__arm_inout("za") int before_attr; // expected-error {{'__arm_inout' only applies to function types}}
int __arm_inout("za") between_attr; // expected-error {{'__arm_inout' only applies to function types}}
const __arm_inout("za") int between_attr_2 = 0; // expected-error {{'__arm_inout' cannot appear here}}
int after_attr __arm_inout("za"); // expected-error {{'__arm_inout' only applies to function types}}
int * __arm_inout("za") ptr_attr; // expected-error {{'__arm_inout' only applies to function types}}
int & __arm_inout("za") ref_attr = after_attr; // expected-error {{'__arm_inout' only applies to function types}}
int && __arm_inout("za") rref_attr = 0; // expected-error {{'__arm_inout' only applies to function types}}
int array_attr [1] __arm_inout("za"); // expected-error {{'__arm_inout' only applies to function types}}
void fn_attr () __arm_inout("za");
void noexcept_fn_attr () noexcept __arm_inout("za");
struct MemberFnOrder {
  virtual void f() const volatile && noexcept __arm_inout("za") final = 0;
};
struct __arm_inout("za") struct_attr; // expected-error {{'__arm_inout' only applies to non-K&R-style functions}}
class __arm_inout("za") class_attr {}; // expected-error {{'__arm_inout' only applies to non-K&R-style functions}}
union __arm_inout("za") union_attr; // expected-error {{'__arm_inout' only applies to non-K&R-style functions}}
enum __arm_inout("za") E { }; // expected-error {{'__arm_inout' only applies to non-K&R-style functions}}
namespace test_misplacement {
__arm_inout("za") struct struct_attr2;  // expected-error {{misplaced '__arm_inout'}}
__arm_inout("za") class class_attr2; // expected-error {{misplaced '__arm_inout'}}
__arm_inout("za") union union_attr2; // expected-error {{misplaced '__arm_inout'}}
__arm_inout("za") enum  E2 { }; // expected-error {{misplaced '__arm_inout'}}
}

// Checks attributes placed at wrong syntactic locations of class specifiers.
class __arm_inout("za") __arm_inout("za") // expected-error 2 {{'__arm_inout' only applies to non-K&R-style functions}}
  attr_after_class_name_decl __arm_inout("za") __arm_inout("za"); // expected-error {{'__arm_inout' cannot appear here}} \
                                                                 expected-error 2 {{'__arm_inout' only applies to non-K&R-style functions}}

class __arm_inout("za") __arm_inout("za") // expected-error 2 {{'__arm_inout' only applies to non-K&R-style functions}}
 attr_after_class_name_definition __arm_inout("za") __arm_inout("za") __arm_inout("za"){}; // expected-error {{'__arm_inout' cannot appear here}} \
                                                                                        expected-error 3 {{'__arm_inout' only applies to non-K&R-style functions}}

class __arm_inout("za") c {}; // expected-error {{'__arm_inout' only applies to non-K&R-style functions}}
class c __arm_inout("za") __arm_inout("za") x; // expected-error 2 {{'__arm_inout' only applies to function types}}
class c __arm_inout("za") __arm_inout("za") y __arm_inout("za") __arm_inout("za"); // expected-error 4 {{'__arm_inout' only applies to function types}}
class c final [(int){0}];

class base {};
class __arm_inout("za") __arm_inout("za") final_class // expected-error 2 {{'__arm_inout' only applies to non-K&R-style functions}}
  __arm_inout("za") alignas(float) final // expected-error {{'__arm_inout' cannot appear here}} \
                                          expected-error {{'__arm_inout' only applies to non-K&R-style functions}}
  __arm_inout("za") alignas(float) __arm_inout("za") alignas(float): base{}; // expected-error {{'__arm_inout' cannot appear here}}

class __arm_inout("za") __arm_inout("za") final_class_another // expected-error 2 {{'__arm_inout' only applies to non-K&R-style functions}}
  __arm_inout("za") __arm_inout("za") alignas(16) final // expected-error {{'__arm_inout' cannot appear here}} \
                                                       expected-error 2 {{'__arm_inout' only applies to non-K&R-style functions}}
  __arm_inout("za") __arm_inout("za") alignas(16) __arm_inout("za"){}; // expected-error {{'__arm_inout' cannot appear here}}

class after_class_close {} __arm_inout("za"); // expected-error {{'__arm_inout' cannot appear here, place it after "class" to apply it to the type declaration}}

class C {};

__arm_inout("za") struct with_init_declarators {} init_declarator; // expected-error {{'__arm_inout' only applies to function types}}
__arm_inout("za") struct no_init_declarators; // expected-error {{misplaced '__arm_inout'}}
template<typename> __arm_inout("za") struct no_init_declarators_template; // expected-error {{'__arm_inout' cannot appear here}}
void fn_with_structs() {
  __arm_inout("za") struct with_init_declarators {} init_declarator; // expected-error {{'__arm_inout' only applies to function types}}
  __arm_inout("za") struct no_init_declarators; // expected-error {{'__arm_inout' cannot appear here}}
}
__arm_inout("za"); // expected-error {{'__arm_inout' only applies to non-K&R-style functions}}
struct ctordtor {
  __arm_inout("za") ctordtor __arm_inout("za") () __arm_inout("za"); // expected-error 2 {{'__arm_inout' cannot be applied to a declaration}}
  ctordtor (C) __arm_inout("za");
  __arm_inout("za") ~ctordtor __arm_inout("za") () __arm_inout("za"); // expected-error 2 {{'__arm_inout' cannot be applied to a declaration}}
};
__arm_inout("za") ctordtor::ctordtor __arm_inout("za") () __arm_inout("za") {} // expected-error 2 {{'__arm_inout' cannot be applied to a declaration}}
__arm_inout("za") ctordtor::ctordtor (C) __arm_inout("za") try {} catch (...) {} // expected-error {{'__arm_inout' cannot be applied to a declaration}}
__arm_inout("za") ctordtor::~ctordtor __arm_inout("za") () __arm_inout("za") {} // expected-error 2 {{'__arm_inout' cannot be applied to a declaration}}
extern "C++" __arm_inout("za") int extern_attr; // expected-error {{'__arm_inout' only applies to function types}}
template <typename T> __arm_inout("za") void template_attr (); // expected-error {{'__arm_inout' cannot be applied to a declaration}}
__arm_inout("za") __arm_inout("za") int __arm_inout("za") __arm_inout("za") multi_attr __arm_inout("za") __arm_inout("za"); // expected-error 6 {{'__arm_inout' only applies to function types}}

int (paren_attr) __arm_inout("za"); // expected-error {{'__arm_inout' cannot appear here}}
unsigned __arm_inout("za") int attr_in_decl_spec; // expected-error {{'__arm_inout' cannot appear here}}
unsigned __arm_inout("za") int __arm_inout("za") const double_decl_spec = 0; // expected-error 2 {{'__arm_inout' cannot appear here}}
class foo {
  void const_after_attr () __arm_inout("za") const; // expected-error {{expected ';'}}
};
extern "C++" __arm_inout("za") { } // expected-error {{'__arm_inout' cannot appear here}}
__arm_inout("za") extern "C++" { } // expected-error {{'__arm_inout' cannot appear here}}
__arm_inout("za") template <typename T> void before_template_attr (); // expected-error {{'__arm_inout' cannot appear here}}
__arm_inout("za") namespace ns { int i; } // expected-error {{'__arm_inout' cannot appear here}}
__arm_inout("za") static_assert(true, ""); //expected-error {{'__arm_inout' cannot appear here}}
__arm_inout("za") asm(""); // expected-error {{'__arm_inout' cannot appear here}}

__arm_inout("za") using ns::i; // expected-warning {{ISO C++}} \
                                expected-error {{'__arm_inout' only applies to non-K&R-style functions}}
__arm_inout("za") using namespace ns; // expected-error {{'__arm_inout' only applies to non-K&R-style functions}}
namespace __arm_inout("za") ns2 {} // expected-warning {{attributes on a namespace declaration are a C++17 extension}} \
                                    expected-error {{'__arm_inout' only applies to non-K&R-style functions}}

using __arm_inout("za") alignas(4)__arm_inout("za") ns::i;          // expected-warning 2 {{ISO C++}} \
                                                                   expected-error {{'__arm_inout' cannot appear here}} \
                                                                   expected-error {{'alignas' attribute only applies to variables, data members and tag types}} \
                                                                   expected-warning {{ISO C++}} \
                                                                   expected-error 2 {{'__arm_inout' only applies to non-K&R-style functions}}
using __arm_inout("za") alignas(4) __arm_inout("za") foobar = int; // expected-error {{'__arm_inout' cannot appear here}} \
                                                                  expected-error {{'alignas' attribute only applies to}} \
                                                                  expected-error 2 {{'__arm_inout' only applies to function types}}

__arm_inout("za") using T = int; // expected-error {{'__arm_inout' cannot appear here}}
using T __arm_inout("za") = int; // expected-error {{'__arm_inout' only applies to function types}}
template<typename T> using U __arm_inout("za") = T; // expected-error {{'__arm_inout' only applies to function types}}
using ns::i __arm_inout("za"); // expected-warning {{ISO C++}} \
                                expected-error {{'__arm_inout' only applies to non-K&R-style functions}}
using ns::i __arm_inout("za"), ns::i __arm_inout("za"); // expected-warning 2 {{ISO C++}} \
                                                       expected-warning {{use of multiple declarators in a single using declaration is a C++17 extension}} \
                                                       expected-error 2 {{'__arm_inout' only applies to non-K&R-style functions}}
struct using_in_struct_base {
  typedef int i, j, k, l;
};
struct using_in_struct : using_in_struct_base {
  __arm_inout("za") using using_in_struct_base::i; // expected-warning {{ISO C++}} \
                                                    expected-error {{'__arm_inout' only applies to non-K&R-style functions}}
  using using_in_struct_base::j __arm_inout("za"); // expected-warning {{ISO C++}} \
                                                    expected-error {{'__arm_inout' only applies to non-K&R-style functions}}
  __arm_inout("za") using using_in_struct_base::k __arm_inout("za"), using_in_struct_base::l __arm_inout("za"); // expected-warning 3 {{ISO C++}} \
                                                                                                             expected-warning {{use of multiple declarators in a single using declaration is a C++17 extension}} \
                                                                                                             expected-error 4 {{'__arm_inout' only applies to non-K&R-style functions}}
};
using __arm_inout("za") ns::i; // expected-warning {{ISO C++}} \
                                expected-error {{'__arm_inout' cannot appear here}} \
                                expected-error {{'__arm_inout' only applies to non-K&R-style functions}}
using T __arm_inout("za") = int; // expected-error {{'__arm_inout' only applies to function types}}

auto trailing() -> __arm_inout("za") const int; // expected-error {{'__arm_inout' cannot appear here}}
auto trailing() -> const __arm_inout("za") int; // expected-error {{'__arm_inout' cannot appear here}}
auto trailing() -> const int __arm_inout("za"); // expected-error {{'__arm_inout' only applies to function types}}
auto trailing_2() -> struct struct_attr __arm_inout("za"); // expected-error {{'__arm_inout' only applies to function types}}

namespace N {
  struct S {};
};
template<typename> struct Template {};

// FIXME: Improve this diagnostic
struct __arm_inout("za") N::S s; // expected-error {{'__arm_inout' cannot appear here}}
struct __arm_inout("za") Template<int> t; // expected-error {{'__arm_inout' cannot appear here}}
struct __arm_inout("za") ::template Template<int> u; // expected-error {{'__arm_inout' cannot appear here}}
template struct __arm_inout("za") Template<char>; // expected-error {{'__arm_inout' cannot appear here}}
template struct __attribute__((pure)) Template<std::size_t>; // We still allow GNU-style attributes here
template <> struct __arm_inout("za") Template<void>; // expected-error {{'__arm_inout' only applies to non-K&R-style functions}}

enum __arm_inout("za") E1 {}; // expected-error {{'__arm_inout' only applies to non-K&R-style functions}}
enum __arm_inout("za") E2; // expected-error {{forbids forward references}} \
                            expected-error {{'__arm_inout' only applies to non-K&R-style functions}}
enum __arm_inout("za") E1; // expected-error {{'__arm_inout' only applies to non-K&R-style functions}}
enum __arm_inout("za") E3 : int; // expected-error {{'__arm_inout' only applies to non-K&R-style functions}}
enum __arm_inout("za") { // expected-error {{'__arm_inout' only applies to non-K&R-style functions}}
  k_123 __arm_inout("za") = 123 // expected-warning {{attributes on an enumerator declaration are a C++17 extension}} \
                                 expected-error {{'__arm_inout' only applies to non-K&R-style functions}}
};
enum __arm_inout("za") E1 e; // expected-error {{'__arm_inout' cannot appear here}}
enum __arm_inout("za") class E4 { }; // expected-error {{'__arm_inout' cannot appear here}}
enum struct __arm_inout("za") E5; // expected-error {{'__arm_inout' only applies to non-K&R-style functions}}
enum E6 {} __arm_inout("za"); // expected-error {{'__arm_inout' cannot appear here, place it after "enum" to apply it to the type declaration}}

struct S {
  friend int f __arm_inout("za") (); // expected-error {{'__arm_inout' cannot appear here}} \
                                      expected-error {{'__arm_inout' cannot be applied to a declaration}}
  friend int f2 __arm_inout("za") () {} // expected-error {{'__arm_inout' cannot be applied to a declaration}}
  __arm_inout("za") friend int g(); // expected-error {{'__arm_inout' cannot appear here}}
  __arm_inout("za") friend int h() { // expected-error {{'__arm_inout' cannot be applied to a declaration}}
  }
  __arm_inout("za") friend int f3(), f4(), f5(); // expected-error {{'__arm_inout' cannot appear here}}
  friend int f6 __arm_inout("za") (), f7 __arm_inout("za") (), f8 __arm_inout("za") (); // expected-error3 {{'__arm_inout' cannot appear here}} \
                                                                                     expected-error 3 {{'__arm_inout' cannot be applied to a declaration}}
  friend class __arm_inout("za") C; // expected-error {{'__arm_inout' cannot appear here}}
  __arm_inout("za") friend class D; // expected-error {{'__arm_inout' cannot appear here}}
  __arm_inout("za") friend int; // expected-error {{'__arm_inout' cannot appear here}}
};
template<typename T> void tmpl (T) {}
template __arm_inout("za") void tmpl(char); // expected-error {{'__arm_inout' cannot appear here}}
template void __arm_inout("za") tmpl(short); // expected-error {{'__arm_inout' only applies to function types}}

// Statement tests
void foo () {
  __arm_inout("za") ; // expected-error {{'__arm_inout' cannot be applied to a statement}}
  __arm_inout("za") { } // expected-error {{'__arm_inout' cannot be applied to a statement}}
  __arm_inout("za") if (0) { } // expected-error {{'__arm_inout' cannot be applied to a statement}}
  __arm_inout("za") for (;;); // expected-error {{'__arm_inout' cannot be applied to a statement}}
  __arm_inout("za") do { // expected-error {{'__arm_inout' cannot be applied to a statement}}
    __arm_inout("za") continue; // expected-error {{'__arm_inout' cannot be applied to a statement}}
  } while (0);
  __arm_inout("za") while (0); // expected-error {{'__arm_inout' cannot be applied to a statement}}

  __arm_inout("za") switch (i) { // expected-error {{'__arm_inout' cannot be applied to a statement}}
    __arm_inout("za") case 0: // expected-error {{'__arm_inout' cannot be applied to a statement}}
    __arm_inout("za") default: // expected-error {{'__arm_inout' cannot be applied to a statement}}
      __arm_inout("za") break; // expected-error {{'__arm_inout' cannot be applied to a statement}}
  }

  __arm_inout("za") goto there; // expected-error {{'__arm_inout' cannot be applied to a statement}}
  __arm_inout("za") there: // expected-error {{'__arm_inout' only applies to non-K&R-style functions}}

  __arm_inout("za") try { // expected-error {{'__arm_inout' cannot be applied to a statement}}
  } __arm_inout("za") catch (...) { // expected-error {{'__arm_inout' cannot appear here}}
  }

  void bar __arm_inout("za") (__arm_inout("za") int i, __arm_inout("za") int j); // expected-error 2 {{'__arm_inout' only applies to function types}} \
                                                                              expected-error {{'__arm_inout' cannot be applied to a declaration}}
  using FuncType = void (__arm_inout("za") int); // expected-error {{'__arm_inout' only applies to function types}}
  void baz(__arm_inout("za")...); // expected-error {{expected parameter declarator}}

  __arm_inout("za") return; // expected-error {{'__arm_inout' cannot be applied to a statement}}
}

// Expression tests
void bar () {
  new int[42]__arm_inout("za")[5]__arm_inout("za"){}; // expected-error {{'__arm_inout' only applies to function types}}
}

// Condition tests
void baz () {
  if (__arm_inout("za") bool b = true) { // expected-error {{'__arm_inout' only applies to function types}}
    switch (__arm_inout("za") int n { 42 }) { // expected-error {{'__arm_inout' only applies to function types}}
    default:
      for (__arm_inout("za") int n = 0; __arm_inout("za") char b = n < 5; ++b) { // expected-error 2 {{'__arm_inout' only applies to function types}}
      }
    }
  }
  int x;
  // An attribute can be applied to an expression-statement, such as the first
  // statement in a for. But it can't be applied to a condition which is an
  // expression.
  for (__arm_inout("za") x = 0; ; ) {} // expected-error {{'__arm_inout' cannot appear here}}
  for (; __arm_inout("za") x < 5; ) {} // expected-error {{'__arm_inout' cannot appear here}}
  while (__arm_inout("za") bool k { false }) { // expected-error {{'__arm_inout' only applies to function types}}
  }
  while (__arm_inout("za") true) { // expected-error {{'__arm_inout' cannot appear here}}
  }
  do {
  } while (__arm_inout("za") false); // expected-error {{'__arm_inout' cannot appear here}}

  for (__arm_inout("za") int n : { 1, 2, 3 }) { // expected-error {{'__arm_inout' only applies to function types}}
  }
}

enum class __attribute__((visibility("hidden"))) SecretKeepers {
  one, /* rest are deprecated */ two, three
};
enum class __arm_inout("za") EvenMoreSecrets {}; // expected-error {{'__arm_inout' only applies to non-K&R-style functions}}

// Forbid attributes on decl specifiers.
unsigned __arm_inout("za") static int __arm_inout("za") v1; // expected-error {{'__arm_inout' only applies to function types}} \
           expected-error {{'__arm_inout' cannot appear here}}
typedef __arm_inout("za") unsigned long __arm_inout("za") v2; // expected-error {{'__arm_inout' only applies to function types}} \
          expected-error {{'__arm_inout' cannot appear here}}
int __arm_inout("za") foo(int __arm_inout("za") x); // expected-error 2 {{'__arm_inout' only applies to function types}}

__arm_inout("za"); // expected-error {{'__arm_inout' only applies to non-K&R-style functions}}

class A {
  A(__arm_inout("za") int a); // expected-error {{'__arm_inout' only applies to function types}}
};
A::A(__arm_inout("za") int a) {} // expected-error {{'__arm_inout' only applies to function types}}

template<typename T> struct TemplateStruct {};
class FriendClassesWithAttributes {
  // We allow GNU-style attributes here
  template <class _Tp, class _Alloc> friend class __attribute__((__type_visibility__("default"))) vector;
  template <class _Tp, class _Alloc> friend class __declspec(code_seg("foo,whatever")) vector2;
  // But not C++11 ones
  template <class _Tp, class _Alloc> friend class __arm_inout("za") vector3;                                         // expected-error {{'__arm_inout' cannot appear here}}

  // Also allowed
  friend struct __attribute__((__type_visibility__("default"))) TemplateStruct<FriendClassesWithAttributes>;
  friend struct __declspec(code_seg("foo,whatever")) TemplateStruct<FriendClassesWithAttributes>;
  friend struct __arm_inout("za") TemplateStruct<FriendClassesWithAttributes>;                                       // expected-error {{'__arm_inout' cannot appear here}}
};

// Check ordering: C++11 attributes must appear before GNU attributes.
class Ordering {
  void f1(
    int (__arm_inout("za") __attribute__(()) int n) // expected-error {{'__arm_inout' only applies to function types}}
  ) {
  }

  void f2(
      int (*)(__arm_inout("za") __attribute__(()) int n) // expected-error {{'__arm_inout' only applies to function types}}
  ) {
  }

  void f3(
    int (__attribute__(()) __arm_inout("za") int n) // expected-error {{'__arm_inout' cannot appear here}}
  ) {
  }

  void f4(
      int (*)(__attribute__(()) __arm_inout("za") int n) // expected-error {{'__arm_inout' cannot appear here}}
  ) {
  }
};

namespace base_specs {
struct A {};
struct B : __arm_inout("za") A {}; // expected-error {{'__arm_inout' cannot be applied to a base specifier}}
struct C : __arm_inout("za") virtual A {}; // expected-error {{'__arm_inout' cannot be applied to a base specifier}}
struct D : __arm_inout("za") public virtual A {}; // expected-error {{'__arm_inout' cannot be applied to a base specifier}}
struct E : public __arm_inout("za") virtual A {}; // expected-error {{'__arm_inout' cannot appear here}} \
                                                   expected-error {{'__arm_inout' cannot be applied to a base specifier}}
struct F : virtual __arm_inout("za") public A {}; // expected-error {{'__arm_inout' cannot appear here}} \
                                                   expected-error {{'__arm_inout' cannot be applied to a base specifier}}
}

namespace __arm_inout("za") ns_attr {}; // expected-error {{'__arm_inout' only applies to non-K&R-style functions}} \
                                         expected-warning {{attributes on a namespace declaration are a C++17 extension}}
