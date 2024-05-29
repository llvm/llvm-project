// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.4-library -finclude-default-header -verify %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.4-library -finclude-default-header -verify -fnative-half-type %s
// RUN: %clang_cc1 -triple spirv-linux-vulkan-library -finclude-default-header -verify %s
// RUN: %clang_cc1 -triple spirv-linux-vulkan-library -finclude-default-header -verify -fnative-half-type %s

// This test is adapted from the test in DXC:
// tools/clang/test/SemaHLSL/v202x/conforming-literals/valid-literals.hlsl

template <typename T, typename U>
struct is_same {
  static const bool value = false;
};

template <typename T>
struct is_same<T, T> {
  static const bool value = true;
};

bool B; // Used for ternary operator tests below

////////////////////////////////////////////////////////////////////////////////
// Literals Without Suffixes
////////////////////////////////////////////////////////////////////////////////

_Static_assert(is_same<__decltype(1.0), float>::value, "Literals are now float");

_Static_assert(is_same<__decltype(0), int>::value, "0 is int");
_Static_assert(is_same<__decltype(1), int>::value, "1 is int");

// Decimal literals are always signed.
_Static_assert(is_same<__decltype(2147483647), int>::value, "2147483647 is int");
_Static_assert(is_same<__decltype(2147483648), int64_t>::value, "2147483648 is int64_t");
_Static_assert(is_same<__decltype(4294967296), int64_t>::value, "4294967296 is int64_t");

// This is an anomaly that exists in C as well as HLSL. This value can't be
// represented as a signed integer, but base-10 literals are always signed.
// Clang emits a warning that it is interpreting it as unsigned because that is
// not conforming to the C standard.

// expected-warning@+1{{integer literal is too large to be represented in type 'long' and is subject to undefined behavior under C++98, interpreting as 'unsigned long'; this literal will be ill-formed in C++11 onwards}}
static const uint64_t V = 9223372036854775808;

_Static_assert(is_same<__decltype(0x0), int>::value, "0x0 is int");
_Static_assert(is_same<__decltype(0x70000000), int>::value, "0x70000000 is int");
_Static_assert(is_same<__decltype(0xF0000000), uint>::value, "0xF0000000 is uint");

_Static_assert(is_same<__decltype(0x7000000000000000), int64_t>::value, "0x7000000000000000 is int64_t");
_Static_assert(is_same<__decltype(0xF000000000000000), uint64_t>::value, "0xF000000000000000 is uint64_t");

////////////////////////////////////////////////////////////////////////////////
// Integer literals With Suffixes
////////////////////////////////////////////////////////////////////////////////

_Static_assert(is_same<__decltype(1l), int64_t>::value, "1l is int64_t");
_Static_assert(is_same<__decltype(1ul), uint64_t>::value, "1ul is uint64_t");
_Static_assert(is_same<__decltype(1lu), uint64_t>::value, "1lu is uint64_t");

// HLSL 2021 does not define a `long long` type, so the suffix should be
// invalid.
_Static_assert(is_same<__decltype(1ll), int64_t>::value, "1ll is int64_t");
_Static_assert(is_same<__decltype(1ull), uint64_t>::value, "1ull is uint64_t");
_Static_assert(is_same<__decltype(1llu), uint64_t>::value, "1llu is uint64_t");

// Verify that the size of `long long` is the same as the size of `int64_t`.
_Static_assert(sizeof(__decltype(1ll)) == sizeof(int64_t), "sizeof(1ll) == sizeof(int64_t)");
_Static_assert(sizeof(__decltype(1llu)) == sizeof(uint64_t), "sizeof(1llu) == sizeof(uint64_t)");

////////////////////////////////////////////////////////////////////////////////
// Ternary operators on integer literals
////////////////////////////////////////////////////////////////////////////////

_Static_assert(is_same<__decltype(B ? 1 : 1), int>::value, "B ? 1 : 1 is int");

_Static_assert(is_same<__decltype(B ? 1l : 1), int64_t>::value, "B ? 1l : 1 is int64_t");
_Static_assert(is_same<__decltype(B ? 1 : 1l), int64_t>::value, "B ? 1 : 1l is int64_t");

_Static_assert(is_same<__decltype(B ? 1ul : 1), uint64_t>::value, "B ? 1ul : 1 is uint64_t");
_Static_assert(is_same<__decltype(B ? 1 : 1ul), uint64_t>::value, "B ? 1 : 1ul is uint64_t");

////////////////////////////////////////////////////////////////////////////////
// Floating point literals With Suffixes
////////////////////////////////////////////////////////////////////////////////

_Static_assert(is_same<__decltype(1.0h), half>::value, "1.0h is half");
_Static_assert(is_same<__decltype(1.0f), float>::value, "1.0f is float");
_Static_assert(is_same<__decltype(1.0l), double>::value, "1.0l is double");

////////////////////////////////////////////////////////////////////////////////
// Ternary operators on floating point literals
////////////////////////////////////////////////////////////////////////////////

_Static_assert(is_same<__decltype(B ? 1.0 : 1.0), float>::value, "B ? 1.0 : 1.0 is float");

_Static_assert(is_same<__decltype(B ? 1.0l : 1.0l), double>::value, "B ? 1.0l : 1.0l is double");
_Static_assert(is_same<__decltype(B ? 1.0f : 1.0f), float>::value, "B ? 1.0f : 1.0f is float");


_Static_assert(is_same<__decltype(B ? 1.0f : 1.0l), double>::value, "B ? 1.0f : 1.0l is double");
_Static_assert(is_same<__decltype(B ? 1.0l : 1.0f), double>::value, "B ? 1.0l : 1.0f is double");

_Static_assert(is_same<__decltype(B ? 1.0l : 1.0), double>::value, "B ? 1.0l : 1.0 is double");
_Static_assert(is_same<__decltype(B ? 1.0 : 1.0l), double>::value, "B ? 1.0 : 1.0l is double");
_Static_assert(is_same<__decltype(B ? 1.0f : 1.0), float>::value, "B ? 1.0f : 1.0 is float");
_Static_assert(is_same<__decltype(B ? 1.0 : 1.0f), float>::value, "B ? 1.0 : 1.0f is float");

_Static_assert(is_same<__decltype(B ? 1.0h : 1.0h), half>::value, "B ? 1.0h : 1.0h is half");

_Static_assert(is_same<__decltype(B ? 1.0f : 1.0h), float>::value, "B ? 1.0f : 1.0h is float");
_Static_assert(is_same<__decltype(B ? 1.0h : 1.0f), float>::value, "B ? 1.0h : 1.0f is float");

_Static_assert(is_same<__decltype(B ? 1.0l : 1.0h), double>::value, "B ? 1.0l : 1.0h is double");
_Static_assert(is_same<__decltype(B ? 1.0h : 1.0l), double>::value, "B ? 1.0h : 1.0l is double");

_Static_assert(is_same<__decltype(B ? 1.0h : 1.0), float>::value, "B ? 1.0h : 1.0 is float");
_Static_assert(is_same<__decltype(B ? 1.0 : 1.0h), float>::value, "B ? 1.0 : 1.0h is float");
