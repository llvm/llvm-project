// RUN: %clang_cc1 -fcxx-exceptions -fdeclspec -fexceptions -fsyntax-only -verify -std=c++11 -Wc++14-compat -Wc++14-extensions -Wc++17-extensions %s

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
[[]] int before_attr;
int [[]] between_attr;
const [[]] int between_attr_2 = 0; // expected-error {{an attribute list cannot appear here}}
int after_attr [[]];
int * [[]] ptr_attr;
int & [[]] ref_attr = after_attr;
int & [[unknown]] ref_attr_2 = after_attr; // expected-warning {{unknown attribute 'unknown' ignored}}
int & [[noreturn]] ref_attr_3 = after_attr; // expected-error {{'noreturn' attribute cannot be applied to types}}
int && [[]] rref_attr = 0;
int array_attr [1] [[]];
alignas(8) int aligned_attr;
[[test::valid(for 42 [very] **** '+' symbols went on a trip and had a "good"_time; the end.)]] int garbage_attr; // expected-warning {{unknown attribute 'valid' ignored}}
[[,,,static, class, namespace,, inline, constexpr, mutable,, bitand, bitor::compl(!.*_ Cx.!U^*R),,,]] int more_garbage_attr; // expected-warning {{unknown attribute 'static' ignored}} \
    // expected-warning {{unknown attribute 'class' ignored}} \
    // expected-warning {{unknown attribute 'namespace' ignored}} \
    // expected-warning {{unknown attribute 'inline' ignored}} \
    // expected-warning {{unknown attribute 'constexpr' ignored}} \
    // expected-warning {{unknown attribute 'mutable' ignored}} \
    // expected-warning {{unknown attribute 'bitand' ignored}} \
    // expected-warning {{unknown attribute 'compl' ignored}}
[[u8"invalid!"]] int invalid_string_attr; // expected-error {{expected ']'}}
void fn_attr () [[]];
void noexcept_fn_attr () noexcept [[]];
struct MemberFnOrder {
  virtual void f() const volatile && noexcept [[]] final = 0;
};
struct [[]] struct_attr;
class [[]] class_attr {};
union [[]] union_attr;
enum [[]] E { };
namespace test_misplacement {
[[]] struct struct_attr2;  //expected-error{{misplaced attributes}}
[[]] class class_attr2; //expected-error{{misplaced attributes}}
[[]] union union_attr2; //expected-error{{misplaced attributes}}
[[]] enum  E2 { }; //expected-error{{misplaced attributes}}
}

// Checks attributes placed at wrong syntactic locations of class specifiers.
class [[]] [[]]
  attr_after_class_name_decl [[]] [[]]; // expected-error {{an attribute list cannot appear here}}

class [[]] [[]]
 attr_after_class_name_definition [[]] [[]] [[]]{}; // expected-error {{an attribute list cannot appear here}}

class [[]] c {};
class c [[]] [[]] x;
class c [[]] [[]] y [[]] [[]];
class c final [(int){0}];

class base {};
class [[]] [[]] final_class
  alignas(float) [[]] final // expected-error {{an attribute list cannot appear here}}
  alignas(float) [[]] [[]] alignas(float): base{}; // expected-error {{an attribute list cannot appear here}}

class [[]] [[]] final_class_another
  [[]] [[]] alignas(16) final // expected-error {{an attribute list cannot appear here}}
  [[]] [[]] alignas(16) [[]]{}; // expected-error {{an attribute list cannot appear here}}

// The diagnostics here don't matter much, this just shouldn't crash:
class C final [[deprecated(l]] {}); // expected-error {{use of undeclared identifier}} expected-error {{expected ']'}} expected-error {{an attribute list cannot appear here}} expected-error {{expected unqualified-id}}
class D final alignas ([l) {}]{}); // expected-error {{expected ',' or ']' in lambda capture list}} expected-error {{an attribute list cannot appear here}}

[[]] struct with_init_declarators {} init_declarator;
[[]] struct no_init_declarators; // expected-error {{misplaced attributes}}
template<typename> [[]] struct no_init_declarators_template; // expected-error {{an attribute list cannot appear here}}
void fn_with_structs() {
  [[]] struct with_init_declarators {} init_declarator;
  [[]] struct no_init_declarators; // expected-error {{an attribute list cannot appear here}}
}
[[]];
struct ctordtor {
  [[]] ctordtor [[]] () [[]];
  ctordtor (C) [[]];
  [[]] ~ctordtor [[]] () [[]];
};
[[]] ctordtor::ctordtor [[]] () [[]] {}
[[]] ctordtor::ctordtor (C) [[]] try {} catch (...) {}
[[]] ctordtor::~ctordtor [[]] () [[]] {}
extern "C++" [[]] int extern_attr;
template <typename T> [[]] void template_attr ();
[[]] [[]] int [[]] [[]] multi_attr [[]] [[]];

int comma_attr [[,]];
int scope_attr [[foo::]]; // expected-error {{expected identifier}}
int (paren_attr) [[]]; // expected-error {{an attribute list cannot appear here}}
unsigned [[]] int attr_in_decl_spec; // expected-error {{an attribute list cannot appear here}}
unsigned [[]] int [[]] const double_decl_spec = 0; // expected-error 2{{an attribute list cannot appear here}}
class foo {
  void const_after_attr () [[]] const; // expected-error {{expected ';'}}
};
extern "C++" [[]] { } // expected-error {{an attribute list cannot appear here}}
[[]] template <typename T> void before_template_attr (); // expected-error {{an attribute list cannot appear here}}
[[]] namespace ns { int i; } // expected-error {{an attribute list cannot appear here}} expected-note {{declared here}}
[[]] static_assert(true, ""); //expected-error {{an attribute list cannot appear here}}
[[]] asm(""); // expected-error {{an attribute list cannot appear here}}

[[]] using ns::i; // expected-error {{an attribute list cannot appear here}}
[[unknown]] using namespace ns; // expected-warning {{unknown attribute 'unknown' ignored}}
[[noreturn]] using namespace ns; // expected-error {{'noreturn' attribute only applies to functions}}
namespace [[]] ns2 {} // expected-warning {{attributes on a namespace declaration are a C++17 extension}}

using [[]] alignas(4) [[]] ns::i; // expected-error {{an attribute list cannot appear here}}
using [[]] alignas(4) [[]] foobar = int; // expected-error {{an attribute list cannot appear here}} expected-error {{'alignas' attribute only applies to}}

void bad_attributes_in_do_while() {
  do {} while (
      [[ns::i); // expected-error {{expected ']'}} \
                // expected-note {{to match this '['}} \
                // expected-error {{expected expression}}
  do {} while (
      [[a]b ns::i); // expected-error {{expected ']'}} \
                    // expected-note {{to match this '['}} \
                    // expected-error {{expected expression}}
  do {} while (
      [[ab]ab] ns::i); // expected-error {{an attribute list cannot appear here}}
  do {} while ( // expected-note {{to match this '('}}
      alignas(4 ns::i; // expected-note {{to match this '('}}
} // expected-error 2{{expected ')'}} expected-error {{expected expression}}

[[]] using T = int; // expected-error {{an attribute list cannot appear here}}
using T [[]] = int; // ok
template<typename T> using U [[]] = T;
using ns::i [[]]; // expected-error {{an attribute list cannot appear here}}
using [[]] ns::i; // expected-error {{an attribute list cannot appear here}}
using T [[unknown]] = int; // expected-warning {{unknown attribute 'unknown' ignored}}
using T [[noreturn]] = int; // expected-error {{'noreturn' attribute only applies to functions}}
using V = int; // expected-note {{previous}}
using V [[gnu::vector_size(16)]] = int; // expected-error {{redefinition with different types}}

auto trailing() -> [[]] const int; // expected-error {{an attribute list cannot appear here}}
auto trailing() -> const [[]] int; // expected-error {{an attribute list cannot appear here}}
auto trailing() -> const int [[]];
auto trailing_2() -> struct struct_attr [[]];

namespace N {
  struct S {};
};
template<typename> struct Template {};

// FIXME: Improve this diagnostic
struct [[]] N::S s; // expected-error {{an attribute list cannot appear here}}
struct [[]] Template<int> t; // expected-error {{an attribute list cannot appear here}}
struct [[]] ::template Template<int> u; // expected-error {{an attribute list cannot appear here}}
template struct [[]] Template<char>; // expected-error {{an attribute list cannot appear here}}
template struct __attribute__((pure)) Template<std::size_t>; // We still allow GNU-style attributes here
template <> struct [[]] Template<void>;

enum [[]] E1 {};
enum [[]] E2; // expected-error {{forbids forward references}}
enum [[]] E1;
enum [[]] E3 : int;
enum [[]] {
  k_123 [[]] = 123 // expected-warning {{attributes on an enumerator declaration are a C++17 extension}}
};
enum [[]] E1 e; // expected-error {{an attribute list cannot appear here}}
enum [[]] class E4 { }; // expected-error {{an attribute list cannot appear here}}
enum struct [[]] E5;

struct S {
  friend int f [[]] (); // expected-FIXME{{an attribute list cannot appear here}}
  friend int f1 [[noreturn]] (); //expected-error{{an attribute list cannot appear here}}
  friend int f2 [[]] [[noreturn]] () {}
  [[]] friend int g(); // expected-error{{an attribute list cannot appear here}}
  [[]] friend int h() {
  }
  [[]] friend int f3(), f4(), f5(); // expected-error{{an attribute list cannot appear here}}
  friend int f6 [[noreturn]] (), f7 [[noreturn]] (), f8 [[noreturn]] (); // expected-error3 {{an attribute list cannot appear here}}
  friend class [[]] C; // expected-error{{an attribute list cannot appear here}}
  [[]] friend class D; // expected-error{{an attribute list cannot appear here}}
  [[]] friend int; // expected-error{{an attribute list cannot appear here}}
};
template<typename T> void tmpl(T) {}
template void tmpl [[]] (int); // expected-FIXME {{an attribute list cannot appear here}}
template [[]] void tmpl(char); // expected-error {{an attribute list cannot appear here}}
template void [[]] tmpl(short);

// Argument tests
alignas int aligned_no_params; // expected-error {{expected '('}}
alignas(i) int aligned_nonconst; // expected-error {{'aligned' attribute requires integer constant}} expected-note {{read of non-const variable 'i'}}

// Statement tests
void foo () {
  [[]] ;
  [[]] { }
  [[]] if (0) { }
  [[]] for (;;);
  [[]] do {
    [[]] continue;
  } while (0);
  [[]] while (0);

  [[]] switch (i) {
    [[]] case 0:
    [[]] default:
      [[]] break;
  }

  [[]] goto there;
  [[]] there:

  [[]] try {
  } [[]] catch (...) { // expected-error {{an attribute list cannot appear here}}
  }
  struct S { int arr[2]; } s;
  (void)s.arr[ [] { return 0; }() ]; // expected-error {{C++11 only allows consecutive left square brackets when introducing an attribute}}
  int n = __builtin_offsetof(S, arr[ [] { return 0; }() ]); // expected-error {{C++11 only allows consecutive left square brackets when introducing an attribute}}

  void bar [[noreturn]] ([[]] int i, [[]] int j);
  using FuncType = void ([[]] int);
  void baz([[]]...); // expected-error {{expected parameter declarator}}

  [[]] return;
}

template<typename...Ts> void variadic() {
  void bar [[noreturn...]] (); // expected-error {{attribute 'noreturn' cannot be used as an attribute pack}}
}

// Expression tests
void bar () {
  // FIXME: GCC accepts [[gnu::noreturn]] on a lambda, even though it appertains
  // to the operator()'s type, and GCC does not otherwise accept attributes
  // applied to types. Use that to test this.
  [] () [[gnu::noreturn]] { return; } (); // expected-warning {{attribute 'noreturn' ignored}} FIXME-error {{should not return}}
  [] () [[gnu::noreturn]] { throw; } (); // expected-warning {{attribute 'noreturn' ignored}}
  new int[42][[]][5][[]]{};
}

// Condition tests
void baz () {
  if ([[unknown]] bool b = true) { // expected-warning {{unknown attribute 'unknown' ignored}}
    switch ([[unknown]] int n { 42 }) { // expected-warning {{unknown attribute 'unknown' ignored}}
    default:
      for ([[unknown]] int n = 0; [[unknown]] char b = n < 5; ++b) { // expected-warning 2{{unknown attribute 'unknown' ignored}}
      }
    }
  }
  int x;
  // An attribute can be applied to an expression-statement, such as the first
  // statement in a for. But it can't be applied to a condition which is an
  // expression.
  for ([[]] x = 0; ; ) {} // expected-error {{an attribute list cannot appear here}}
  for (; [[]] x < 5; ) {} // expected-error {{an attribute list cannot appear here}}
  while ([[]] bool k { false }) {
  }
  while ([[]] true) { // expected-error {{an attribute list cannot appear here}}
  }
  do {
  } while ([[]] false); // expected-error {{an attribute list cannot appear here}}

  for ([[unknown]] int n : { 1, 2, 3 }) { // expected-warning {{unknown attribute 'unknown' ignored}}
  }
}

enum class __attribute__((visibility("hidden"))) SecretKeepers {
  one, /* rest are deprecated */ two, three
};
enum class [[]] EvenMoreSecrets {};

namespace arguments {
  void f[[gnu::format(printf, 1, 2)]](const char*, ...);
  void g() [[unknown::foo(ignore arguments for unknown attributes, even with symbols!)]]; // expected-warning {{unknown attribute 'foo' ignored}}
  [[deprecated("with argument")]] int i;
  // expected-warning@-1 {{use of the 'deprecated' attribute is a C++14 extension}}
}

// Forbid attributes on decl specifiers.
unsigned [[gnu::used]] static int [[gnu::unused]] v1; // expected-error {{'unused' attribute cannot be applied to types}} \
           expected-error {{an attribute list cannot appear here}}
typedef [[gnu::used]] unsigned long [[gnu::unused]] v2; // expected-error {{'unused' attribute cannot be applied to types}} \
          expected-error {{an attribute list cannot appear here}}
int [[carries_dependency]] foo(int [[carries_dependency]] x); // expected-error 2{{'carries_dependency' attribute cannot be applied to types}}

// Forbid [[gnu::...]] attributes on declarator chunks.
int *[[gnu::unused]] v3; // expected-warning {{attribute 'unused' ignored}}
int v4[2][[gnu::unused]]; // expected-warning {{attribute 'unused' ignored}}
int v5()[[gnu::unused]]; // expected-warning {{attribute 'unused' ignored}}

[[attribute_declaration]]; // expected-warning {{unknown attribute 'attribute_declaration' ignored}}
[[noreturn]]; // expected-error {{'noreturn' attribute only applies to functions}}
[[carries_dependency]]; // expected-error {{'carries_dependency' attribute only applies to parameters, Objective-C methods, and functions}}

class A {
  A([[gnu::unused]] int a);
};
A::A([[gnu::unused]] int a) {}

namespace GccConst {
  // GCC's tokenizer treats const and __const as the same token.
  [[gnu::const]] int *f1();
  [[gnu::__const]] int *f2();
  [[gnu::__const__]] int *f3();
  void f(const int *);
  void g() { f(f1()); f(f2()); }
  void h() { f(f3()); }
}

namespace GccASan {
  __attribute__((no_address_safety_analysis)) void f1();
  __attribute__((no_sanitize_address)) void f2();
  [[gnu::no_address_safety_analysis]] void f3();
  [[gnu::no_sanitize_address]] void f4();
}

namespace {
  [[deprecated]] void bar();
  // expected-warning@-1 {{use of the 'deprecated' attribute is a C++14 extension}}
  [[deprecated("hello")]] void baz();
  // expected-warning@-1 {{use of the 'deprecated' attribute is a C++14 extension}}
  [[deprecated()]] void foo();
  // expected-error@-1 {{parentheses must be omitted if 'deprecated' attribute's argument list is empty}}
  [[gnu::deprecated()]] void quux();
}

namespace {
[[ // expected-error {{expected ']'}}
#pragma pack(pop)
deprecated
]] void bad();
}

int fallthru(int n) {
  switch (n) {
  case 0:
    n += 5;
    [[fallthrough]]; // expected-warning {{use of the 'fallthrough' attribute is a C++17 extension}}
  case 1:
    n *= 2;
    break;
  }
  return n;
}

template<typename T> struct TemplateStruct {};
class FriendClassesWithAttributes {
  // We allow GNU-style attributes here
  template <class _Tp, class _Alloc> friend class __attribute__((__type_visibility__("default"))) vector;
  template <class _Tp, class _Alloc> friend class __declspec(code_seg("whatever")) vector2;
  // But not C++11 ones
  template <class _Tp, class _Alloc> friend class[[]] vector3;                                         // expected-error {{an attribute list cannot appear here}}
  template <class _Tp, class _Alloc> friend class [[clang::__type_visibility__(("default"))]] vector4; // expected-error {{an attribute list cannot appear here}}

  // Also allowed
  friend struct __attribute__((__type_visibility__("default"))) TemplateStruct<FriendClassesWithAttributes>;
  friend struct __declspec(code_seg("whatever")) TemplateStruct<FriendClassesWithAttributes>;
  friend struct[[]] TemplateStruct<FriendClassesWithAttributes>;                                       // expected-error {{an attribute list cannot appear here}}
  friend struct [[clang::__type_visibility__("default")]] TemplateStruct<FriendClassesWithAttributes>; // expected-error {{an attribute list cannot appear here}}
};

#define attr_name bitand
#define attr_name_2(x) x
#define attr_name_3(x, y) x##y
[[attr_name, attr_name_2(bitor), attr_name_3(com, pl)]] int macro_attrs; // expected-warning {{unknown attribute 'compl' ignored}} \
   expected-warning {{unknown attribute 'bitor' ignored}} \
   expected-warning {{unknown attribute 'bitand' ignored}}

// Check that we can parse an attribute in our vendor namespace.
[[clang::annotate("test")]] void annotate1();
[[_Clang::annotate("test")]] void annotate2();
// Note: __clang__ is a predefined macro, which is why _Clang is the
// prefered "protected" vendor namespace. We support __clang__ only for
// people expecting it to behave the same as __gnu__.
[[__clang__::annotate("test")]] void annotate3();  // expected-warning {{'__clang__' is a predefined macro name, not an attribute scope specifier; did you mean '_Clang' instead?}}
