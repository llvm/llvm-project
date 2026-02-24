// RUN: %clang_cc1 -fsyntax-only -verify -triple x86_64-unknown-linux-gnu -std=c++20 %s
// expected-no-diagnostics

// --- Type classification traits ---

static_assert(__is_integral(__int256));
static_assert(__is_integral(unsigned __int256));
static_assert(__is_integral(__int256_t));
static_assert(__is_integral(__uint256_t));

static_assert(__is_arithmetic(__int256));
static_assert(__is_arithmetic(unsigned __int256));

static_assert(__is_scalar(__int256));
static_assert(__is_scalar(unsigned __int256));

static_assert(__is_fundamental(__int256));
static_assert(__is_fundamental(unsigned __int256));

// --- Signedness traits ---

static_assert(__is_signed(__int256));
static_assert(!__is_unsigned(__int256));
static_assert(!__is_signed(unsigned __int256));
static_assert(__is_unsigned(unsigned __int256));

static_assert(__is_signed(__int256_t));
static_assert(__is_unsigned(__uint256_t));

// --- __builtin_is_implicit_lifetime ---

static_assert(__builtin_is_implicit_lifetime(__int256));
static_assert(__builtin_is_implicit_lifetime(unsigned __int256));

// --- __make_signed / __make_unsigned ---

static_assert(__is_same(__make_signed(__int256), __int256));
static_assert(__is_same(__make_signed(unsigned __int256), __int256));
static_assert(__is_same(__make_unsigned(__int256), unsigned __int256));
static_assert(__is_same(__make_unsigned(unsigned __int256), unsigned __int256));

// With cv-qualifiers
static_assert(__is_same(__make_signed(const __int256), const __int256));
static_assert(__is_same(__make_signed(volatile unsigned __int256), volatile __int256));
static_assert(__is_same(__make_signed(const volatile unsigned __int256), const volatile __int256));
static_assert(__is_same(__make_unsigned(const __int256), const unsigned __int256));
static_assert(__is_same(__make_unsigned(volatile __int256), volatile unsigned __int256));

// --- Enum with __int256 underlying type ---

enum E256 : __int256_t { E256_Zero = 0, E256_One = 1 };
enum U256 : __uint256_t { U256_Zero = 0, U256_One = 1 };

static_assert(__is_same(__make_signed(E256), __int256_t));
static_assert(__is_same(__make_unsigned(E256), __uint256_t));
static_assert(__is_same(__make_signed(U256), __int256_t));
static_assert(__is_same(__make_unsigned(U256), __uint256_t));

// --- sizeof / alignof ---

static_assert(sizeof(__int256) == 32);
static_assert(alignof(__int256) == 16);
static_assert(sizeof(unsigned __int256) == 32);
static_assert(alignof(unsigned __int256) == 16);
static_assert(sizeof(__int256_t) == 32);
static_assert(sizeof(__uint256_t) == 32);

// --- Overload resolution ---

constexpr int select_overload(__int128) { return 128; }
constexpr int select_overload(__int256_t) { return 256; }

static_assert(select_overload((__int256_t)0) == 256);
static_assert(select_overload((__int128)0) == 128);
