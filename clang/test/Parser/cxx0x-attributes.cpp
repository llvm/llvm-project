// RUN: %clang_cc1 -fcxx-exceptions -fexceptions -fsyntax-only -verify -std=c++11 %s

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
[[]] struct with_init_declarators {} init_declarator;
[[]] struct no_init_declarators; // expected-error {{an attribute list cannot appear here}}
[[]];
struct ctordtor {
  [[]] ctordtor();
  [[]] ~ctordtor();
};
[[]] ctordtor::ctordtor() {}
[[]] ctordtor::~ctordtor() {}
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
[[]] using namespace ns;

[[]] using T = int; // expected-error {{an attribute list cannot appear here}}
using T [[]] = int; // ok
template<typename T> using U [[]] = T;
using ns::i [[]]; // expected-error {{an attribute list cannot appear here}}
using [[]] ns::i; // expected-error {{an attribute list cannot appear here}}

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
template <> struct [[]] Template<void>;

enum [[]] E1 {};
enum [[]] E2; // expected-error {{forbids forward references}}
enum [[]] E1;
enum [[]] E3 : int;
enum [[]] {
  k_123 [[]] = 123 // expected-error {{an attribute list cannot appear here}}
};
enum [[]] E1 e; // expected-error {{an attribute list cannot appear here}}
enum [[]] class E4 { }; // expected-error {{an attribute list cannot appear here}}
enum struct [[]] E5;

struct S {
  friend int f [[]] (); // expected-FIXME{{an attribute list cannot appear here}}
  [[]] friend int g(); // expected-FIXME{{an attribute list cannot appear here}}
  [[]] friend int h() {
  }
  friend class [[]] C; // expected-error{{an attribute list cannot appear here}}
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
  [] () [[noreturn]] { return; } (); // expected-error {{should not return}}
  [] () [[noreturn]] { throw; } ();
  new int[42][[]][5][[]]{};
}

// Condition tests
void baz () {
  if ([[]] bool b = true) {
    switch ([[]] int n { 42 }) {
    default:
      for ([[]] int n = 0; [[]] char b = n < 5; ++b) {
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

  for ([[]] int n : { 1, 2, 3 }) {
  }
}

enum class __attribute__((visibility("hidden"))) SecretKeepers {
  one, /* rest are deprecated */ two, three
};
enum class [[]] EvenMoreSecrets {};

namespace arguments {
  // FIXME: remove the sema warnings after migrating existing gnu attributes to c++11 syntax.
  void f(const char*, ...) [[gnu::format(printf, 1, 2)]]; // expected-warning {{unknown attribute 'format' ignored}}
  void g() [[unknown::foo(currently arguments of attributes from unknown namespace other than 'gnu' namespace are ignored... blah...)]]; // expected-warning {{unknown attribute 'foo' ignored}}
}

// forbid attributes on decl specifiers
unsigned [[gnu::used]] static int [[gnu::unused]] v1; // expected-warning {{attribute 'unused' ignored, because it is not attached to a declaration}} \
           expected-error {{an attribute list cannot appear here}}
typedef [[gnu::used]] unsigned long [[gnu::unused]] v2; // expected-warning {{attribute 'unused' ignored, because it is not attached to a declaration}} \
          expected-error {{an attribute list cannot appear here}}
int [[carries_dependency]] foo(int [[carries_dependency]] x); // expected-warning 2{{attribute 'carries_dependency' ignored, because it is not attached to a declaration}}
