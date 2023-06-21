// RUN: %clang_cc1 -fcxx-exceptions -fdeclspec -fexceptions -fsyntax-only -verify -std=c++11 -Wc++14-compat -Wc++14-extensions -Wc++17-extensions -triple aarch64-none-linux-gnu %s

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
__arm_streaming int before_attr; // expected-error {{'__arm_streaming' only applies to function types}}
int __arm_streaming between_attr; // expected-error {{'__arm_streaming' only applies to function types}}
const __arm_streaming int between_attr_2 = 0; // expected-error {{'__arm_streaming' cannot appear here}}
int after_attr __arm_streaming; // expected-error {{'__arm_streaming' only applies to function types}}
int * __arm_streaming ptr_attr; // expected-error {{'__arm_streaming' only applies to function types}}
int & __arm_streaming ref_attr = after_attr; // expected-error {{'__arm_streaming' only applies to function types}}
int && __arm_streaming rref_attr = 0; // expected-error {{'__arm_streaming' only applies to function types}}
int array_attr [1] __arm_streaming; // expected-error {{'__arm_streaming' only applies to function types}}
void fn_attr () __arm_streaming;
void noexcept_fn_attr () noexcept __arm_streaming;
struct MemberFnOrder {
  virtual void f() const volatile && noexcept __arm_streaming final = 0;
};
struct __arm_streaming struct_attr; // expected-error {{'__arm_streaming' cannot be applied to a declaration}}
class __arm_streaming class_attr {}; // expected-error {{'__arm_streaming' cannot be applied to a declaration}}
union __arm_streaming union_attr; // expected-error {{'__arm_streaming' cannot be applied to a declaration}}
enum __arm_streaming E { }; // expected-error {{'__arm_streaming' cannot be applied to a declaration}}
namespace test_misplacement {
__arm_streaming struct struct_attr2;  // expected-error {{misplaced '__arm_streaming'}}
__arm_streaming class class_attr2; // expected-error {{misplaced '__arm_streaming'}}
__arm_streaming union union_attr2; // expected-error {{misplaced '__arm_streaming'}}
__arm_streaming enum  E2 { }; // expected-error {{misplaced '__arm_streaming'}}
}

// Checks attributes placed at wrong syntactic locations of class specifiers.
class __arm_streaming __arm_streaming // expected-error 2 {{'__arm_streaming' cannot be applied to a declaration}}
  attr_after_class_name_decl __arm_streaming __arm_streaming; // expected-error {{'__arm_streaming' cannot appear here}} \
                                                                 expected-error 2 {{'__arm_streaming' cannot be applied to a declaration}}

class __arm_streaming __arm_streaming // expected-error 2 {{'__arm_streaming' cannot be applied to a declaration}}
 attr_after_class_name_definition __arm_streaming __arm_streaming __arm_streaming{}; // expected-error {{'__arm_streaming' cannot appear here}} \
                                                                                        expected-error 3 {{'__arm_streaming' cannot be applied to a declaration}}

class __arm_streaming c {}; // expected-error {{'__arm_streaming' cannot be applied to a declaration}}
class c __arm_streaming __arm_streaming x; // expected-error 2 {{'__arm_streaming' only applies to function types}}
class c __arm_streaming __arm_streaming y __arm_streaming __arm_streaming; // expected-error 4 {{'__arm_streaming' only applies to function types}}
class c final [(int){0}];

class base {};
class __arm_streaming __arm_streaming final_class // expected-error 2 {{'__arm_streaming' cannot be applied to a declaration}}
  __arm_streaming alignas(float) final // expected-error {{'__arm_streaming' cannot appear here}} \
                                          expected-error {{'__arm_streaming' cannot be applied to a declaration}}
  __arm_streaming alignas(float) __arm_streaming alignas(float): base{}; // expected-error {{'__arm_streaming' cannot appear here}}

class __arm_streaming __arm_streaming final_class_another // expected-error 2 {{'__arm_streaming' cannot be applied to a declaration}}
  __arm_streaming __arm_streaming alignas(16) final // expected-error {{'__arm_streaming' cannot appear here}} \
                                                       expected-error 2 {{'__arm_streaming' cannot be applied to a declaration}}
  __arm_streaming __arm_streaming alignas(16) __arm_streaming{}; // expected-error {{'__arm_streaming' cannot appear here}}

class after_class_close {} __arm_streaming; // expected-error {{'__arm_streaming' cannot appear here, place it after "class" to apply it to the type declaration}}

class C {};

__arm_streaming struct with_init_declarators {} init_declarator; // expected-error {{'__arm_streaming' only applies to function types}}
__arm_streaming struct no_init_declarators; // expected-error {{misplaced '__arm_streaming'}}
template<typename> __arm_streaming struct no_init_declarators_template; // expected-error {{'__arm_streaming' cannot appear here}}
void fn_with_structs() {
  __arm_streaming struct with_init_declarators {} init_declarator; // expected-error {{'__arm_streaming' only applies to function types}}
  __arm_streaming struct no_init_declarators; // expected-error {{'__arm_streaming' cannot appear here}}
}
__arm_streaming; // expected-error {{'__arm_streaming' cannot be applied to a declaration}}
struct ctordtor {
  __arm_streaming ctordtor __arm_streaming () __arm_streaming; // expected-error 2 {{'__arm_streaming' cannot be applied to a declaration}}
  ctordtor (C) __arm_streaming;
  __arm_streaming ~ctordtor __arm_streaming () __arm_streaming; // expected-error 2 {{'__arm_streaming' cannot be applied to a declaration}}
};
__arm_streaming ctordtor::ctordtor __arm_streaming () __arm_streaming {} // expected-error 2 {{'__arm_streaming' cannot be applied to a declaration}}
__arm_streaming ctordtor::ctordtor (C) __arm_streaming try {} catch (...) {} // expected-error {{'__arm_streaming' cannot be applied to a declaration}}
__arm_streaming ctordtor::~ctordtor __arm_streaming () __arm_streaming {} // expected-error 2 {{'__arm_streaming' cannot be applied to a declaration}}
extern "C++" __arm_streaming int extern_attr; // expected-error {{'__arm_streaming' only applies to function types}}
template <typename T> __arm_streaming void template_attr (); // expected-error {{'__arm_streaming' cannot be applied to a declaration}}
__arm_streaming __arm_streaming int __arm_streaming __arm_streaming multi_attr __arm_streaming __arm_streaming; // expected-error 6 {{'__arm_streaming' only applies to function types}}

int (paren_attr) __arm_streaming; // expected-error {{'__arm_streaming' cannot appear here}}
unsigned __arm_streaming int attr_in_decl_spec; // expected-error {{'__arm_streaming' cannot appear here}}
unsigned __arm_streaming int __arm_streaming const double_decl_spec = 0; // expected-error 2 {{'__arm_streaming' cannot appear here}}
class foo {
  void const_after_attr () __arm_streaming const; // expected-error {{expected ';'}}
};
extern "C++" __arm_streaming { } // expected-error {{'__arm_streaming' cannot appear here}}
__arm_streaming extern "C++" { } // expected-error {{'__arm_streaming' cannot appear here}}
__arm_streaming template <typename T> void before_template_attr (); // expected-error {{'__arm_streaming' cannot appear here}}
__arm_streaming namespace ns { int i; } // expected-error {{'__arm_streaming' cannot appear here}}
__arm_streaming static_assert(true, ""); //expected-error {{'__arm_streaming' cannot appear here}}
__arm_streaming asm(""); // expected-error {{'__arm_streaming' cannot appear here}}

__arm_streaming using ns::i; // expected-warning {{ISO C++}} \
                                expected-error {{'__arm_streaming' cannot be applied to a declaration}}
__arm_streaming using namespace ns; // expected-error {{'__arm_streaming' cannot be applied to a declaration}}
namespace __arm_streaming ns2 {} // expected-warning {{attributes on a namespace declaration are a C++17 extension}} \
                                    expected-error {{'__arm_streaming' cannot be applied to a declaration}}

using __arm_streaming alignas(4)__arm_streaming ns::i;          // expected-warning 2 {{ISO C++}} \
                                                                   expected-error {{'__arm_streaming' cannot appear here}} \
                                                                   expected-error {{'alignas' attribute only applies to variables, data members and tag types}} \
                                                                   expected-warning {{ISO C++}} \
                                                                   expected-error 2 {{'__arm_streaming' cannot be applied to a declaration}}
using __arm_streaming alignas(4) __arm_streaming foobar = int; // expected-error {{'__arm_streaming' cannot appear here}} \
                                                                  expected-error {{'alignas' attribute only applies to}} \
                                                                  expected-error 2 {{'__arm_streaming' only applies to function types}}

__arm_streaming using T = int; // expected-error {{'__arm_streaming' cannot appear here}}
using T __arm_streaming = int; // expected-error {{'__arm_streaming' only applies to function types}}
template<typename T> using U __arm_streaming = T; // expected-error {{'__arm_streaming' only applies to function types}}
using ns::i __arm_streaming; // expected-warning {{ISO C++}} \
                                expected-error {{'__arm_streaming' cannot be applied to a declaration}}
using ns::i __arm_streaming, ns::i __arm_streaming; // expected-warning 2 {{ISO C++}} \
                                                       expected-warning {{use of multiple declarators in a single using declaration is a C++17 extension}} \
                                                       expected-error 2 {{'__arm_streaming' cannot be applied to a declaration}}
struct using_in_struct_base {
  typedef int i, j, k, l;
};
struct using_in_struct : using_in_struct_base {
  __arm_streaming using using_in_struct_base::i; // expected-warning {{ISO C++}} \
                                                    expected-error {{'__arm_streaming' cannot be applied to a declaration}}
  using using_in_struct_base::j __arm_streaming; // expected-warning {{ISO C++}} \
                                                    expected-error {{'__arm_streaming' cannot be applied to a declaration}}
  __arm_streaming using using_in_struct_base::k __arm_streaming, using_in_struct_base::l __arm_streaming; // expected-warning 3 {{ISO C++}} \
                                                                                                             expected-warning {{use of multiple declarators in a single using declaration is a C++17 extension}} \
                                                                                                             expected-error 4 {{'__arm_streaming' cannot be applied to a declaration}}
};
using __arm_streaming ns::i; // expected-warning {{ISO C++}} \
                                expected-error {{'__arm_streaming' cannot appear here}} \
                                expected-error {{'__arm_streaming' cannot be applied to a declaration}}
using T __arm_streaming = int; // expected-error {{'__arm_streaming' only applies to function types}}

auto trailing() -> __arm_streaming const int; // expected-error {{'__arm_streaming' cannot appear here}}
auto trailing() -> const __arm_streaming int; // expected-error {{'__arm_streaming' cannot appear here}}
auto trailing() -> const int __arm_streaming; // expected-error {{'__arm_streaming' only applies to function types}}
auto trailing_2() -> struct struct_attr __arm_streaming; // expected-error {{'__arm_streaming' only applies to function types}}

namespace N {
  struct S {};
};
template<typename> struct Template {};

// FIXME: Improve this diagnostic
struct __arm_streaming N::S s; // expected-error {{'__arm_streaming' cannot appear here}}
struct __arm_streaming Template<int> t; // expected-error {{'__arm_streaming' cannot appear here}}
struct __arm_streaming ::template Template<int> u; // expected-error {{'__arm_streaming' cannot appear here}}
template struct __arm_streaming Template<char>; // expected-error {{'__arm_streaming' cannot appear here}}
template struct __attribute__((pure)) Template<std::size_t>; // We still allow GNU-style attributes here
template <> struct __arm_streaming Template<void>; // expected-error {{'__arm_streaming' cannot be applied to a declaration}}

enum __arm_streaming E1 {}; // expected-error {{'__arm_streaming' cannot be applied to a declaration}}
enum __arm_streaming E2; // expected-error {{forbids forward references}} \
                            expected-error {{'__arm_streaming' cannot be applied to a declaration}}
enum __arm_streaming E1; // expected-error {{'__arm_streaming' cannot be applied to a declaration}}
enum __arm_streaming E3 : int; // expected-error {{'__arm_streaming' cannot be applied to a declaration}}
enum __arm_streaming { // expected-error {{'__arm_streaming' cannot be applied to a declaration}}
  k_123 __arm_streaming = 123 // expected-warning {{attributes on an enumerator declaration are a C++17 extension}} \
                                 expected-error {{'__arm_streaming' cannot be applied to a declaration}}
};
enum __arm_streaming E1 e; // expected-error {{'__arm_streaming' cannot appear here}}
enum __arm_streaming class E4 { }; // expected-error {{'__arm_streaming' cannot appear here}}
enum struct __arm_streaming E5; // expected-error {{'__arm_streaming' cannot be applied to a declaration}}
enum E6 {} __arm_streaming; // expected-error {{'__arm_streaming' cannot appear here, place it after "enum" to apply it to the type declaration}}

struct S {
  friend int f __arm_streaming (); // expected-error {{'__arm_streaming' cannot appear here}} \
                                      expected-error {{'__arm_streaming' cannot be applied to a declaration}}
  friend int f2 __arm_streaming () {} // expected-error {{'__arm_streaming' cannot be applied to a declaration}}
  __arm_streaming friend int g(); // expected-error {{'__arm_streaming' cannot appear here}}
  __arm_streaming friend int h() { // expected-error {{'__arm_streaming' cannot be applied to a declaration}}
  }
  __arm_streaming friend int f3(), f4(), f5(); // expected-error {{'__arm_streaming' cannot appear here}}
  friend int f6 __arm_streaming (), f7 __arm_streaming (), f8 __arm_streaming (); // expected-error3 {{'__arm_streaming' cannot appear here}} \
                                                                                     expected-error 3 {{'__arm_streaming' cannot be applied to a declaration}}
  friend class __arm_streaming C; // expected-error {{'__arm_streaming' cannot appear here}}
  __arm_streaming friend class D; // expected-error {{'__arm_streaming' cannot appear here}}
  __arm_streaming friend int; // expected-error {{'__arm_streaming' cannot appear here}}
};
template<typename T> void tmpl (T) {}
template __arm_streaming void tmpl(char); // expected-error {{'__arm_streaming' cannot appear here}}
template void __arm_streaming tmpl(short); // expected-error {{'__arm_streaming' only applies to function types}}

// Statement tests
void foo () {
  __arm_streaming ; // expected-error {{'__arm_streaming' cannot be applied to a statement}}
  __arm_streaming { } // expected-error {{'__arm_streaming' cannot be applied to a statement}}
  __arm_streaming if (0) { } // expected-error {{'__arm_streaming' cannot be applied to a statement}}
  __arm_streaming for (;;); // expected-error {{'__arm_streaming' cannot be applied to a statement}}
  __arm_streaming do { // expected-error {{'__arm_streaming' cannot be applied to a statement}}
    __arm_streaming continue; // expected-error {{'__arm_streaming' cannot be applied to a statement}}
  } while (0);
  __arm_streaming while (0); // expected-error {{'__arm_streaming' cannot be applied to a statement}}

  __arm_streaming switch (i) { // expected-error {{'__arm_streaming' cannot be applied to a statement}}
    __arm_streaming case 0: // expected-error {{'__arm_streaming' cannot be applied to a statement}}
    __arm_streaming default: // expected-error {{'__arm_streaming' cannot be applied to a statement}}
      __arm_streaming break; // expected-error {{'__arm_streaming' cannot be applied to a statement}}
  }

  __arm_streaming goto there; // expected-error {{'__arm_streaming' cannot be applied to a statement}}
  __arm_streaming there: // expected-error {{'__arm_streaming' cannot be applied to a declaration}}

  __arm_streaming try { // expected-error {{'__arm_streaming' cannot be applied to a statement}}
  } __arm_streaming catch (...) { // expected-error {{'__arm_streaming' cannot appear here}}
  }

  void bar __arm_streaming (__arm_streaming int i, __arm_streaming int j); // expected-error 2 {{'__arm_streaming' only applies to function types}} \
                                                                              expected-error {{'__arm_streaming' cannot be applied to a declaration}}
  using FuncType = void (__arm_streaming int); // expected-error {{'__arm_streaming' only applies to function types}}
  void baz(__arm_streaming...); // expected-error {{expected parameter declarator}}

  __arm_streaming return; // expected-error {{'__arm_streaming' cannot be applied to a statement}}
}

// Expression tests
void bar () {
  new int[42]__arm_streaming[5]__arm_streaming{}; // expected-error {{'__arm_streaming' only applies to function types}}
}

// Condition tests
void baz () {
  if (__arm_streaming bool b = true) { // expected-error {{'__arm_streaming' only applies to function types}}
    switch (__arm_streaming int n { 42 }) { // expected-error {{'__arm_streaming' only applies to function types}}
    default:
      for (__arm_streaming int n = 0; __arm_streaming char b = n < 5; ++b) { // expected-error 2 {{'__arm_streaming' only applies to function types}}
      }
    }
  }
  int x;
  // An attribute can be applied to an expression-statement, such as the first
  // statement in a for. But it can't be applied to a condition which is an
  // expression.
  for (__arm_streaming x = 0; ; ) {} // expected-error {{'__arm_streaming' cannot appear here}}
  for (; __arm_streaming x < 5; ) {} // expected-error {{'__arm_streaming' cannot appear here}}
  while (__arm_streaming bool k { false }) { // expected-error {{'__arm_streaming' only applies to function types}}
  }
  while (__arm_streaming true) { // expected-error {{'__arm_streaming' cannot appear here}}
  }
  do {
  } while (__arm_streaming false); // expected-error {{'__arm_streaming' cannot appear here}}

  for (__arm_streaming int n : { 1, 2, 3 }) { // expected-error {{'__arm_streaming' only applies to function types}}
  }
}

enum class __attribute__((visibility("hidden"))) SecretKeepers {
  one, /* rest are deprecated */ two, three
};
enum class __arm_streaming EvenMoreSecrets {}; // expected-error {{'__arm_streaming' cannot be applied to a declaration}}

// Forbid attributes on decl specifiers.
unsigned __arm_streaming static int __arm_streaming v1; // expected-error {{'__arm_streaming' only applies to function types}} \
           expected-error {{'__arm_streaming' cannot appear here}}
typedef __arm_streaming unsigned long __arm_streaming v2; // expected-error {{'__arm_streaming' only applies to function types}} \
          expected-error {{'__arm_streaming' cannot appear here}}
int __arm_streaming foo(int __arm_streaming x); // expected-error 2 {{'__arm_streaming' only applies to function types}}

__arm_streaming; // expected-error {{'__arm_streaming' cannot be applied to a declaration}}

class A {
  A(__arm_streaming int a); // expected-error {{'__arm_streaming' only applies to function types}}
};
A::A(__arm_streaming int a) {} // expected-error {{'__arm_streaming' only applies to function types}}

template<typename T> struct TemplateStruct {};
class FriendClassesWithAttributes {
  // We allow GNU-style attributes here
  template <class _Tp, class _Alloc> friend class __attribute__((__type_visibility__("default"))) vector;
  template <class _Tp, class _Alloc> friend class __declspec(code_seg("foo,whatever")) vector2;
  // But not C++11 ones
  template <class _Tp, class _Alloc> friend class __arm_streaming vector3;                                         // expected-error {{'__arm_streaming' cannot appear here}}

  // Also allowed
  friend struct __attribute__((__type_visibility__("default"))) TemplateStruct<FriendClassesWithAttributes>;
  friend struct __declspec(code_seg("foo,whatever")) TemplateStruct<FriendClassesWithAttributes>;
  friend struct __arm_streaming TemplateStruct<FriendClassesWithAttributes>;                                       // expected-error {{'__arm_streaming' cannot appear here}}
};

// Check ordering: C++11 attributes must appear before GNU attributes.
class Ordering {
  void f1(
    int (__arm_streaming __attribute__(()) int n) // expected-error {{'__arm_streaming' only applies to function types}}
  ) {
  }

  void f2(
      int (*)(__arm_streaming __attribute__(()) int n) // expected-error {{'__arm_streaming' only applies to function types}}
  ) {
  }

  void f3(
    int (__attribute__(()) __arm_streaming int n) // expected-error {{'__arm_streaming' cannot appear here}}
  ) {
  }

  void f4(
      int (*)(__attribute__(()) __arm_streaming int n) // expected-error {{'__arm_streaming' cannot appear here}}
  ) {
  }
};

namespace base_specs {
struct A {};
struct B : __arm_streaming A {}; // expected-error {{'__arm_streaming' cannot be applied to a base specifier}}
struct C : __arm_streaming virtual A {}; // expected-error {{'__arm_streaming' cannot be applied to a base specifier}}
struct D : __arm_streaming public virtual A {}; // expected-error {{'__arm_streaming' cannot be applied to a base specifier}}
struct E : public __arm_streaming virtual A {}; // expected-error {{'__arm_streaming' cannot appear here}} \
                                                   expected-error {{'__arm_streaming' cannot be applied to a base specifier}}
struct F : virtual __arm_streaming public A {}; // expected-error {{'__arm_streaming' cannot appear here}} \
                                                   expected-error {{'__arm_streaming' cannot be applied to a base specifier}}
}

namespace __arm_streaming ns_attr {}; // expected-error {{'__arm_streaming' cannot be applied to a declaration}} \
                                         expected-warning {{attributes on a namespace declaration are a C++17 extension}}
