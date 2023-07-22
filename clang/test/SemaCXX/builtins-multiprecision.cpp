// RUN: %clang_cc1 -fsyntax-only -std=c++17 -ffreestanding -verify %s
// expected-no-diagnostics

#include <limits.h>

template<typename T>
struct Result {
  T value;
  T carry;
  constexpr bool operator==(const Result<T> &Other) {
    return value == Other.value && carry == Other.carry;
  }
};

template<typename T>
constexpr Result<T> add(T Lhs, T Rhs, T Carryin)
{
  T Carryout = 0;
  if constexpr(__is_same(T, unsigned char))
    return { __builtin_addcb(Lhs, Rhs, Carryin, &Carryout), Carryout };
  else if constexpr(__is_same(T, unsigned short))
    return { __builtin_addcs(Lhs, Rhs, Carryin, &Carryout), Carryout };
  else if constexpr(__is_same(T, unsigned int))
    return { __builtin_addc(Lhs, Rhs, Carryin, &Carryout), Carryout };
  else if constexpr(__is_same(T, unsigned long))
    return { __builtin_addcl(Lhs, Rhs, Carryin, &Carryout), Carryout };
  else if constexpr(__is_same(T, unsigned long long))
    return { __builtin_addcll(Lhs, Rhs, Carryin, &Carryout), Carryout };
}

static_assert(add<unsigned char>(0, 0, 0) == Result<unsigned char>{0, 0});
static_assert(add<unsigned char>(0, 0, 1) == Result<unsigned char>{1, 0});
static_assert(add<unsigned char>(UCHAR_MAX - 1, 1, 1) == Result<unsigned char>{0, 1});
static_assert(add<unsigned char>(UCHAR_MAX, 1, 0) == Result<unsigned char>{0, 1});
static_assert(add<unsigned char>(UCHAR_MAX, 1, 1) == Result<unsigned char>{1, 1});

static_assert(add<unsigned short>(0, 0, 0) == Result<unsigned short>{0, 0});
static_assert(add<unsigned short>(0, 0, 1) == Result<unsigned short>{1, 0});
static_assert(add<unsigned short>(USHRT_MAX - 1, 1, 1) == Result<unsigned short>{0, 1});
static_assert(add<unsigned short>(USHRT_MAX, 1, 0) == Result<unsigned short>{0, 1});
static_assert(add<unsigned short>(USHRT_MAX, 1, 1) == Result<unsigned short>{1, 1});

static_assert(add<unsigned int>(0, 0, 0) == Result<unsigned int>{0, 0});
static_assert(add<unsigned int>(0, 0, 1) == Result<unsigned int>{1, 0});
static_assert(add<unsigned int>(UINT_MAX - 1, 1, 1) == Result<unsigned int>{0, 1});
static_assert(add<unsigned int>(UINT_MAX, 1, 0) == Result<unsigned int>{0, 1});
static_assert(add<unsigned int>(UINT_MAX, 1, 1) == Result<unsigned int>{1, 1});

static_assert(add<unsigned long>(0, 0, 0) == Result<unsigned long>{0, 0});
static_assert(add<unsigned long>(0, 0, 1) == Result<unsigned long>{1, 0});
static_assert(add<unsigned long>(ULONG_MAX - 1, 1, 1) == Result<unsigned long>{0, 1});
static_assert(add<unsigned long>(ULONG_MAX, 1, 0) == Result<unsigned long>{0, 1});
static_assert(add<unsigned long>(ULONG_MAX, 1, 1) == Result<unsigned long>{1, 1});

static_assert(add<unsigned long long>(0, 0, 0) == Result<unsigned long long>{0, 0});
static_assert(add<unsigned long long>(0, 0, 1) == Result<unsigned long long>{1, 0});
static_assert(add<unsigned long long>(ULLONG_MAX - 1, 1, 1) == Result<unsigned long long>{0, 1});
static_assert(add<unsigned long long>(ULLONG_MAX, 1, 0) == Result<unsigned long long>{0, 1});
static_assert(add<unsigned long long>(ULLONG_MAX, 1, 1) == Result<unsigned long long>{1, 1});

template<typename T>
constexpr Result<T> sub(T Lhs, T Rhs, T Carryin)
{
  T Carryout = 0;
  if constexpr(__is_same(T, unsigned char))
    return { __builtin_subcb(Lhs, Rhs, Carryin, &Carryout), Carryout };
  else if constexpr(__is_same(T, unsigned short))
    return { __builtin_subcs(Lhs, Rhs, Carryin, &Carryout), Carryout };
  else if constexpr(__is_same(T, unsigned int))
    return { __builtin_subc(Lhs, Rhs, Carryin, &Carryout), Carryout };
  else if constexpr(__is_same(T, unsigned long))
    return { __builtin_subcl(Lhs, Rhs, Carryin, &Carryout), Carryout };
  else if constexpr(__is_same(T, unsigned long long))
    return { __builtin_subcll(Lhs, Rhs, Carryin, &Carryout), Carryout };
}

static_assert(sub<unsigned char>(0, 0, 0) == Result<unsigned char>{0, 0});
static_assert(sub<unsigned char>(0, 0, 1) == Result<unsigned char>{UCHAR_MAX, 1});
static_assert(sub<unsigned char>(0, 1, 0) == Result<unsigned char>{UCHAR_MAX, 1});
static_assert(sub<unsigned char>(0, 1, 1) == Result<unsigned char>{UCHAR_MAX - 1, 1});
static_assert(sub<unsigned char>(1, 0, 0) == Result<unsigned char>{1, 0});

static_assert(sub<unsigned short>(0, 0, 0) == Result<unsigned short>{0, 0});
static_assert(sub<unsigned short>(0, 0, 1) == Result<unsigned short>{USHRT_MAX, 1});
static_assert(sub<unsigned short>(0, 1, 0) == Result<unsigned short>{USHRT_MAX, 1});
static_assert(sub<unsigned short>(0, 1, 1) == Result<unsigned short>{USHRT_MAX - 1, 1});
static_assert(sub<unsigned short>(1, 0, 0) == Result<unsigned short>{1, 0});

static_assert(sub<unsigned int>(0, 0, 0) == Result<unsigned int>{0, 0});
static_assert(sub<unsigned int>(0, 0, 1) == Result<unsigned int>{UINT_MAX, 1});
static_assert(sub<unsigned int>(0, 1, 0) == Result<unsigned int>{UINT_MAX, 1});
static_assert(sub<unsigned int>(0, 1, 1) == Result<unsigned int>{UINT_MAX - 1, 1});
static_assert(sub<unsigned int>(1, 0, 0) == Result<unsigned int>{1, 0});

static_assert(sub<unsigned long>(0, 0, 0) == Result<unsigned long>{0, 0});
static_assert(sub<unsigned long>(0, 0, 1) == Result<unsigned long>{ULONG_MAX, 1});
static_assert(sub<unsigned long>(0, 1, 0) == Result<unsigned long>{ULONG_MAX, 1});
static_assert(sub<unsigned long>(0, 1, 1) == Result<unsigned long>{ULONG_MAX - 1, 1});
static_assert(sub<unsigned long>(1, 0, 0) == Result<unsigned long>{1, 0});

static_assert(sub<unsigned long long>(0, 0, 0) == Result<unsigned long long>{0, 0});
static_assert(sub<unsigned long long>(0, 0, 1) == Result<unsigned long long>{ULLONG_MAX, 1});
static_assert(sub<unsigned long long>(0, 1, 0) == Result<unsigned long long>{ULLONG_MAX, 1});
static_assert(sub<unsigned long long>(0, 1, 1) == Result<unsigned long long>{ULLONG_MAX - 1, 1});
static_assert(sub<unsigned long long>(1, 0, 0) == Result<unsigned long long>{1, 0});
