// RUN: %clang_cc1 -std=c++2a -verify %s -fcxx-exceptions -Wno-constant-evaluated -triple=x86_64-linux-gnu

using size_t = decltype(sizeof(int));

namespace std {
inline constexpr bool is_constant_evaluated() noexcept {
  return __builtin_is_constant_evaluated();
}
} // namespace std

extern int dummy; // expected-note 1+ {{declared here}}

static_assert(__builtin_is_constant_evaluated());
static_assert(noexcept(__builtin_is_constant_evaluated()));

constexpr bool b = __builtin_is_constant_evaluated();
static_assert(b);

const int n = __builtin_is_constant_evaluated() ? 4 : dummy;
static_assert(n == 4);
constexpr int cn = __builtin_is_constant_evaluated() ? 11 : dummy;
static_assert(cn == 11);
// expected-error@+1 {{'bn' must be initialized by a constant expression}}
constexpr int bn = __builtin_is_constant_evaluated() ? dummy : 42; // expected-note {{non-const variable 'dummy' is not allowed}}

const int n2 = __builtin_is_constant_evaluated() ? dummy : 42; // expected-note {{declared here}}
static_assert(n2 == 42);                                       // expected-error {{static_assert expression is not an integral constant}}
// expected-note@-1 {{initializer of 'n2' is not a constant expression}}

template <bool V, bool Default = std::is_constant_evaluated()>
struct Templ { static_assert(V); static_assert(Default); };
Templ<__builtin_is_constant_evaluated()> x; // type X<true>

template <class T>
void test_if_constexpr() {
  if constexpr (__builtin_is_constant_evaluated()) {
    static_assert(__is_same(T, int));
  } else {
    using Test = typename T::DOES_NOT_EXIST;
  }
}
template void test_if_constexpr<int>();

void test_array_decl() {
  char x[__builtin_is_constant_evaluated() + std::is_constant_evaluated()];
  static_assert(sizeof(x) == 2, "");
}

void test_case_stmt(int x) {
  switch (x) {
  case 0:                                                                // OK
  case __builtin_is_constant_evaluated():                                // expected-note {{previous case}}
  case std::is_constant_evaluated() + __builtin_is_constant_evaluated(): // expected-note {{previous case}}
  case 1:                                                                // expected-error {{duplicate case value '1'}}
  case 2:                                                                // expected-error {{duplicate case value '2'}}
    break;
  }
}

constexpr size_t good_array_size() {
  return std::is_constant_evaluated() ? 42 : static_cast<size_t>(-1);
}

constexpr size_t bad_array_size() {
  return std::is_constant_evaluated() ? static_cast<size_t>(-1) : 13;
}

template <class T>
constexpr T require_constexpr(T v) {
  if (!std::is_constant_evaluated())
    throw "BOOM";
  return v;
}

void test_new_expr() {
  constexpr size_t TooLarge = -1;
  auto *x = new int[std::is_constant_evaluated() ? 1 : TooLarge];      // expected-error {{array is too large}}
  auto *x2 = new int[std::is_constant_evaluated() ? TooLarge : 1];     // OK
  auto *y = new int[1][std::is_constant_evaluated() ? TooLarge : 1]{}; // expected-error {{array is too large}}
  auto *y2 = new int[1][require_constexpr(42)];
}

void test_alignas_operand() {
  alignas(std::is_constant_evaluated() ? 8 : 2) char dummy;
  static_assert(__alignof(dummy) == 8);
}

void test_static_assert_operand() {
  static_assert(std::is_constant_evaluated(), "");
}

void test_enumerator() {
  enum MyEnum {
    ZERO = 0,
    ONE = std::is_constant_evaluated()
  };
  static_assert(ONE == 1, "");
}

struct TestBitfieldWidth {
  unsigned Bits : std::is_constant_evaluated();
};

void test_operand_of_noexcept_fn() noexcept(std::is_constant_evaluated());
static_assert(noexcept(test_operand_of_noexcept_fn()), "");


namespace test_ref_initialization {
int x;
int y;
int &r = __builtin_is_constant_evaluated() ? x : y;
static_assert(&r == &x);

} // namespace test_ref_initialization

#if defined(__cpp_conditional_explicit)
struct TestConditionalExplicit {
  explicit(!__builtin_is_constant_evaluated()) TestConditionalExplicit(int) {}
};
TestConditionalExplicit e = 42;
#endif
