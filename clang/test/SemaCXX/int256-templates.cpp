// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fsyntax-only -verify %s
//
// Test __int256 behavior with C++ templates, SFINAE, concepts, and conversions.
//
// This exercises advanced C++ interactions that upstream reviewers are likely
// to probe: NTTP (non-type template parameters), SFINAE, implicit/explicit
// conversions, constexpr template metaprogramming, and aggregate initialization.
//
// Uses Clang builtin type traits (__is_integral, etc.) to avoid depending on
// standard library headers, which are not available in %clang_cc1 tests.
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -std=c++20 -fsyntax-only -verify %s
// expected-no-diagnostics

// Minimal enable_if for SFINAE testing without <type_traits>
template <bool B, typename T = void> struct enable_if {};
template <typename T> struct enable_if<true, T> { using type = T; };
template <bool B, typename T = void> using enable_if_t = typename enable_if<B, T>::type;

// Minimal is_same
template <typename T, typename U> struct is_same { static constexpr bool value = false; };
template <typename T> struct is_same<T, T> { static constexpr bool value = true; };

// Minimal conditional
template <bool B, typename T, typename F> struct conditional { using type = F; };
template <typename T, typename F> struct conditional<true, T, F> { using type = T; };
template <bool B, typename T, typename F> using conditional_t = typename conditional<B, T, F>::type;

// ========================================================================
// 1. Non-type template parameter (NTTP)
// ========================================================================

// __int256 can be used as a non-type template parameter in C++20.
template <__int256_t V>
struct IntConstant {
    static constexpr __int256_t value = V;
};

static_assert(IntConstant<0>::value == 0);
static_assert(IntConstant<42>::value == 42);
static_assert(IntConstant<-1>::value == -1);

// Large NTTP value
static_assert(IntConstant<((__int256_t)1 << 200)>::value == ((__int256_t)1 << 200));

// Unsigned NTTP
template <__uint256_t V>
struct UIntConstant {
    static constexpr __uint256_t value = V;
};

static_assert(UIntConstant<0>::value == 0);
static_assert(UIntConstant<~(__uint256_t)0>::value == ~(__uint256_t)0);

// ========================================================================
// 2. SFINAE on __is_integral
// ========================================================================

// Clang builtin __is_integral works for __int256 types.
static_assert(__is_integral(__int256_t));
static_assert(__is_integral(__uint256_t));
static_assert(__is_integral(const __int256_t));
static_assert(__is_integral(volatile __uint256_t));

// SFINAE: enable_if selects the correct overload.
template <typename T, enable_if_t<__is_integral(T)>* = nullptr>
constexpr int classify(T) { return 1; }  // integral

template <typename T, enable_if_t<__is_floating_point(T)>* = nullptr>
constexpr int classify(T) { return 2; }  // floating

static_assert(classify((__int256_t)42) == 1);
static_assert(classify((__uint256_t)42) == 1);
static_assert(classify(3.14) == 2);

// ========================================================================
// 3. Builtin type traits for __int256
// ========================================================================

// __is_signed / __is_unsigned
static_assert(__is_signed(__int256_t));
static_assert(!__is_unsigned(__int256_t));
static_assert(__is_unsigned(__uint256_t));
static_assert(!__is_signed(__uint256_t));

// __is_arithmetic
static_assert(__is_arithmetic(__int256_t));
static_assert(__is_arithmetic(__uint256_t));

// __is_fundamental
static_assert(__is_fundamental(__int256_t));
static_assert(__is_fundamental(__uint256_t));

// __is_scalar
static_assert(__is_scalar(__int256_t));
static_assert(__is_scalar(__uint256_t));

// __is_trivially_copyable
static_assert(__is_trivially_copyable(__int256_t));
static_assert(__is_trivially_copyable(__uint256_t));

// __is_standard_layout
static_assert(__is_standard_layout(__int256_t));
static_assert(__is_standard_layout(__uint256_t));

// __is_trivially_constructible
static_assert(__is_trivially_constructible(__int256_t));
static_assert(__is_trivially_destructible(__int256_t));

// __is_constructible from various integer types
static_assert(__is_constructible(__int256_t, int));
static_assert(__is_constructible(__int256_t, long long));
static_assert(__is_constructible(__int256_t, __int128_t));
static_assert(__is_constructible(__uint256_t, unsigned));
static_assert(__is_constructible(__uint256_t, __uint128_t));

// __is_convertible (implicit conversions)
static_assert(__is_convertible_to(int, __int256_t));
static_assert(__is_convertible_to(__int128_t, __int256_t));
static_assert(__is_convertible_to(__int256_t, __int128_t));

// ========================================================================
// 4. Implicit conversions: __int128 <-> __int256
// ========================================================================

// __int128 -> __int256: implicit widening (no data loss)
constexpr __int256_t widen_s(__int128_t x) { return x; }
constexpr __uint256_t widen_u(__uint128_t x) { return x; }

static_assert(widen_s(42) == 42);
static_assert(widen_s(-1) == -1);
static_assert(widen_u(42) == 42);

// __int256 -> __int128: implicit narrowing (may lose data)
constexpr __int128_t narrow_s(__int256_t x) { return x; }
constexpr __uint128_t narrow_u(__uint256_t x) { return x; }

static_assert(narrow_s(42) == 42);
static_assert(narrow_u(42) == 42);

// int -> __int256: implicit widening
constexpr __int256_t from_int(int x) { return x; }
static_assert(from_int(42) == 42);
static_assert(from_int(-1) == -1);

// ========================================================================
// 5. Template argument deduction
// ========================================================================

template <typename T>
constexpr T identity(T x) { return x; }

static_assert(identity((__int256_t)42) == 42);
static_assert(identity((__uint256_t)42) == 42);

// Deduction with auto
constexpr auto auto_val = (__int256_t)100;
static_assert(is_same<decltype(auto_val), const __int256_t>::value);

// ========================================================================
// 6. constexpr template metaprogramming
// ========================================================================

// Recursive constexpr factorial
template <typename T>
constexpr T factorial(T n) {
    return n <= 1 ? T(1) : n * factorial(n - 1);
}

// 20! = 2432902008176640000 (fits in 64-bit)
static_assert(factorial((__int256_t)20) == 2432902008176640000LL);

// 34! = 295232799039604140847618609643520000000 (doesn't fit in 128-bit)
constexpr __int256_t fact34 = factorial((__int256_t)34);
// Verify lower 64 bits (computed from 34! mod 2^64)
static_assert((unsigned long long)fact34 == 0x445DA75B00000000ULL);

// ========================================================================
// 7. Variadic templates
// ========================================================================

template <typename... Ts>
constexpr auto sum(Ts... args) {
    return (args + ...);
}

static_assert(sum((__int256_t)1, (__int256_t)2, (__int256_t)3) == 6);

// ========================================================================
// 8. Conditional type selection
// ========================================================================

static_assert(sizeof(conditional_t<true, __int256_t, __int128_t>) == 32);
static_assert(sizeof(conditional_t<false, __int256_t, __int128_t>) == 16);

// ========================================================================
// 9. Array and aggregate initialization
// ========================================================================

struct Pair256 {
    __int256_t first;
    __uint256_t second;
};

constexpr Pair256 p = {42, 100};
static_assert(p.first == 42);
static_assert(p.second == 100);

constexpr __int256_t arr[] = {1, 2, 3, 4, 5};
static_assert(arr[0] + arr[4] == 6);

// ========================================================================
// 10. sizeof / alignof
// ========================================================================

static_assert(sizeof(__int256_t) == 32);
static_assert(sizeof(__uint256_t) == 32);
static_assert(alignof(__int256_t) == 16);
static_assert(alignof(__uint256_t) == 16);
static_assert(sizeof(__int256_t) == 2 * sizeof(__int128_t));
