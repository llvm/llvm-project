// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fsyntax-only -verify -DITANIUM %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fsyntax-only -verify -DITANIUM -fexperimental-new-constant-interpreter %s
// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -std=c++20 -fsyntax-only -verify -DMICROSOFT %s
// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -std=c++20 -fsyntax-only -verify -DMICROSOFT -fexperimental-new-constant-interpreter %s

constexpr auto missing = __builtin_type_order(int, long);
// expected-error@-1 {{cannot compute order because type 'std::strong_ordering' was not found; include <compare>}}

namespace std {
  struct strong_ordering {
  enum __order { LT = -1, EQ = 0, GT = 1 };
  __order value;

  constexpr explicit strong_ordering(__order value) : value(value) {}
  constexpr bool operator==(strong_ordering const& other) const {
    return value == other.value;
  }
  static const strong_ordering less;
  static const strong_ordering equal;
  static const strong_ordering greater;
};

inline constexpr strong_ordering strong_ordering::less(__order::LT);
inline constexpr strong_ordering strong_ordering::equal(__order::EQ);
inline constexpr strong_ordering strong_ordering::greater(__order::GT);
} // namespace std

static_assert(__is_same(decltype(__builtin_type_order(int, long)),
                        std::strong_ordering));

static_assert(__builtin_type_order(int, int) == std::strong_ordering::equal);
static_assert(__builtin_type_order(int, int const) != std::strong_ordering::equal);
static_assert(__builtin_type_order(void(*)(int), void(*)(int)) == std::strong_ordering::equal);

static_assert(__builtin_type_order(void*, void const volatile*) != std::strong_ordering::equal);
static_assert(__builtin_type_order(int, long) != std::strong_ordering::equal);
static_assert(__builtin_type_order(int, long).value ==
              -__builtin_type_order(long, int).value);

using int_alias = int;
static_assert(__builtin_type_order(int, int_alias) ==
              std::strong_ordering::equal);

struct incomplete;
static_assert(__builtin_type_order(incomplete, incomplete) ==
              std::strong_ordering::equal);
static_assert(__builtin_type_order(int incomplete::*, char incomplete::*) != std::strong_ordering::equal);
static_assert(__builtin_type_order(void (incomplete::*)(), void (incomplete::*)() const) != std::strong_ordering::equal);

struct A {};
struct B {};
static_assert(__builtin_type_order(A, B) == std::strong_ordering::less);
static_assert(__builtin_type_order(B, A) == std::strong_ordering::greater);
static_assert(__builtin_type_order(void(*)(A), void(*)(B)) == std::strong_ordering::less);
static_assert(__builtin_type_order(A[1], B[1]) == std::strong_ordering::less);

#ifdef ITANIUM
static_assert(__builtin_type_order(A, int) == std::strong_ordering::less);
#endif
#ifdef MICROSOFT
static_assert(__builtin_type_order(A, int) == std::strong_ordering::greater);
#endif

template <class T> struct C;
static_assert(__builtin_type_order(C<A>, C<B>) == std::strong_ordering::less);
static_assert(__builtin_type_order(C<B>, C<A>) == std::strong_ordering::greater);


template <class T, class U>
constexpr std::strong_ordering dependent_order() {
  return __builtin_type_order(T, U);
}
static_assert(dependent_order<int, int>() == std::strong_ordering::equal);
static_assert(dependent_order<const int, const int>() ==
              std::strong_ordering::equal);

template <class T>
constexpr bool dependent_constant_expression() {
  static_assert(__builtin_type_order(T, T) == std::strong_ordering::equal);
  return true;
}
static_assert(dependent_constant_expression<int>());
