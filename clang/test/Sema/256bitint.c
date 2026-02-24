// RUN: %clang_cc1 -fsyntax-only -verify -triple x86_64-apple-darwin9 %s -DHAVE
// RUN: %clang_cc1 -fsyntax-only -verify -triple i686-linux-gnu %s -DHAVE_NOT
// RUN: %clang_cc1 -fsyntax-only -verify -triple aarch64-linux-gnu %s -DHAVE
// RUN: %clang_cc1 -fsyntax-only -verify -triple arm-linux-gnueabi %s -DHAVE_NOT
// RUN: %clang_cc1 -fsyntax-only -verify -triple powerpc64-linux-gnu %s -DHAVE
// RUN: %clang_cc1 -fsyntax-only -verify -triple riscv64-linux-gnu %s -DHAVE
// RUN: %clang_cc1 -fsyntax-only -verify -triple wasm32-unknown-unknown %s -DHAVE_NOT
// RUN: %clang_cc1 -fsyntax-only -verify -triple wasm64-unknown-unknown %s -DHAVE

#ifdef HAVE
// expected-no-diagnostics

// __int256 is supported on all 64-bit targets

__int256_t b256s = (__int256_t)0;
__uint256_t b256u = (__uint256_t)-1;

// Explicit signed/unsigned qualifiers
__int256 i256 = (__int256)0;
signed __int256 si256 = (signed __int256)0;
unsigned __int256 ui256 = (unsigned __int256)-1;

// sizeof / alignof
int sz[sizeof(__int256_t) == 32 ? 1 : -1];
int al[_Alignof(__int256_t) == 16 ? 1 : -1];
int sz2[sizeof(__uint256_t) == 32 ? 1 : -1];
int al2[_Alignof(__uint256_t) == 16 ? 1 : -1];

// __SIZEOF_INT256__ predefined macro
int sizemacro[__SIZEOF_INT256__ == 32 ? 1 : -1];

// Basic arithmetic
__int256_t arith_add(__int256_t a, __int256_t b) { return a + b; }
__int256_t arith_sub(__int256_t a, __int256_t b) { return a - b; }
__int256_t arith_mul(__int256_t a, __int256_t b) { return a * b; }
__int256_t arith_div(__int256_t a, __int256_t b) { return a / b; }
__int256_t arith_rem(__int256_t a, __int256_t b) { return a % b; }

// Bitwise operations (key for Hamming distance / popcount use cases)
__uint256_t bit_and(__uint256_t a, __uint256_t b) { return a & b; }
__uint256_t bit_or(__uint256_t a, __uint256_t b) { return a | b; }
__uint256_t bit_xor(__uint256_t a, __uint256_t b) { return a ^ b; }
__uint256_t bit_not(__uint256_t a) { return ~a; }
__uint256_t bit_shl(__uint256_t a, __uint256_t b) { return a << b; }
__uint256_t bit_shr(__uint256_t a, __uint256_t b) { return a >> b; }

// Comparisons
int cmp_eq(__int256_t a, __int256_t b) { return a == b; }
int cmp_lt(__int256_t a, __int256_t b) { return a < b; }
int cmp_gt(__int256_t a, __int256_t b) { return a > b; }

// Conversions between int256 and int128
__int256_t from128(__int128_t x) { return (__int256_t)x; }
__int128_t to128(__int256_t x) { return (__int128_t)x; }

// Conversion from smaller types
__int256_t from64(long long x) { return (__int256_t)x; }
__uint256_t fromu64(unsigned long long x) { return (__uint256_t)x; }

// Typedef equivalence
typedef __int256_t MyInt256;
MyInt256 typedef_test(MyInt256 a) { return a; }

#else

__int256 n; // expected-error {{__int256 is not supported on this target}}

#if defined(__SIZEOF_INT256__)
#error __SIZEOF_INT256__ should not be defined
#endif

#endif
