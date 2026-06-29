// Tests for the diagnostic implementing the static check proposed by
// P3969R1 ("Fixing std::bit_cast of types with padding bits").
//
// RUN: %clang_cc1 -verify -std=c++20 -fsyntax-only -triple x86_64-pc-linux-gnu %s
// RUN: %clang_cc1 -verify -std=c++20 -fsyntax-only -triple x86_64-pc-linux-gnu -Wno-bit-cast-padding -DNO_WARN %s

#ifdef NO_WARN
// expected-no-diagnostics
#endif

namespace long_double_to_int128 {
#ifndef NO_WARN
// expected-warning@+1 {{is always undefined because it unconditionally maps a padding bit onto a non-padding bit}}
__int128 a = __builtin_bit_cast(__int128, (long double)0);

// expected-warning@+1 {{is always undefined because it unconditionally maps a padding bit onto a non-padding bit}}
unsigned __int128 b = __builtin_bit_cast(unsigned __int128, (long double)0);
#endif
} // namespace long_double_to_int128

namespace bitint_to_int {
#ifndef NO_WARN
// expected-warning@+1 {{is always undefined because it unconditionally maps a padding bit onto a non-padding bit}}
unsigned u = __builtin_bit_cast(unsigned, (_BitInt(31))0);

// expected-warning@+1 {{is always undefined because it unconditionally maps a padding bit onto a non-padding bit}}
unsigned long long ull = __builtin_bit_cast(unsigned long long, (_BitInt(63))0);
#endif
} // namespace bitint_to_int

namespace via_struct {
#ifndef NO_WARN
struct holds_ld { long double v; };
// Mapping padded long double through a struct wrapper is still degenerate.
// expected-warning@+1 {{is always undefined because it unconditionally maps a padding bit onto a non-padding bit}}
__int128 a = __builtin_bit_cast(__int128, holds_ld{0.0L});
#endif
} // namespace via_struct

namespace via_array {
#ifndef NO_WARN
struct two_int128 { __int128 a, b; };
struct ld2 { long double v[2]; };
// expected-warning@+1 {{is always undefined because it unconditionally maps a padding bit onto a non-padding bit}}
two_int128 a = __builtin_bit_cast(two_int128, ld2{});

struct four_int128 { __int128 a, b, c, d; };
struct ld_2x2 { long double v[2][2]; };
// expected-warning@+1 {{is always undefined because it unconditionally maps a padding bit onto a non-padding bit}}
four_int128 b = __builtin_bit_cast(four_int128, ld_2x2{});

struct i128_2 { __int128 v[2]; };
// expected-warning@+1 {{is always undefined because it unconditionally maps a padding bit onto a non-padding bit}}
i128_2 c = __builtin_bit_cast(i128_2, ld2{});
#endif
} // namespace via_array

namespace alignment_padding {
#ifndef NO_WARN
struct internal_pad { char a; int b; };
// expected-warning@+1 {{is always undefined because it unconditionally maps a padding bit onto a non-padding bit}}
unsigned long long a = __builtin_bit_cast(unsigned long long, internal_pad{});

struct trailing_pad { int a; char b; };
// expected-warning@+1 {{is always undefined because it unconditionally maps a padding bit onto a non-padding bit}}
unsigned long long b = __builtin_bit_cast(unsigned long long, trailing_pad{});

struct alignas(sizeof(__int128)) overaligned { long long v; };
// expected-warning@+1 {{is always undefined because it unconditionally maps a padding bit onto a non-padding bit}}
__int128 c = __builtin_bit_cast(__int128, overaligned{});
#endif
} // namespace alignment_padding

namespace empty_struct {
#ifndef NO_WARN
struct empty {};
// expected-warning@+1 {{is always undefined because it unconditionally maps a padding bit onto a non-padding bit}}
signed char from_empty = __builtin_bit_cast(signed char, empty{});

struct wraps_empty { empty e; };
// expected-warning@+1 {{is always undefined because it unconditionally maps a padding bit onto a non-padding bit}}
signed char from_wrapped = __builtin_bit_cast(signed char, wraps_empty{});
#endif

struct empty2 {};
empty2 to_empty = __builtin_bit_cast(empty2, (unsigned char)0);
} // namespace empty_struct

namespace no_warning {
unsigned u1 = __builtin_bit_cast(unsigned, 1);
float f1 = __builtin_bit_cast(float, 1u);

long double ld = __builtin_bit_cast(long double, (unsigned __int128)0);

long double ld2 = __builtin_bit_cast(long double, (long double)0);

struct u4 { unsigned v[4]; };
struct f4 { float v[4]; };
u4 u4_a = __builtin_bit_cast(u4, f4{});
f4 f4_a = __builtin_bit_cast(f4, u4{});

struct bytes16 { unsigned char b[16]; };
bytes16 b16 = __builtin_bit_cast(bytes16, (long double)0);

union U {
  long double ld;
  unsigned char bytes[16];
};
U u = __builtin_bit_cast(U, (long double)0);
__int128 from_union = __builtin_bit_cast(__int128, U{});

// Bit-fields disable the analysis entirely (no false positives).
struct with_bitfield { unsigned long long x : 5; };
with_bitfield wbf = __builtin_bit_cast(with_bitfield, (unsigned long long)0);
} // namespace no_warning

namespace not_byte_like {
#ifndef NO_WARN
struct sc16 { signed char b[16]; };
// expected-warning@+1 {{is always undefined because it unconditionally maps a padding bit onto a non-padding bit}}
sc16 a = __builtin_bit_cast(sc16, (long double)0);

enum my_byte : unsigned char {};
struct mb_array { my_byte b[16]; };
// expected-warning@+1 {{is always undefined because it unconditionally maps a padding bit onto a non-padding bit}}
mb_array mb = __builtin_bit_cast(mb_array, (long double)0);
#endif
} // namespace not_byte_like

namespace templates {
template <class To, class From>
constexpr To my_bit_cast(const From &from) {
  return __builtin_bit_cast(To, from); // #cast
}

#ifndef NO_WARN
// expected-warning@#cast {{is always undefined because it unconditionally maps a padding bit onto a non-padding bit}}
// expected-note@+1 {{in instantiation}}
__int128 instantiated = my_bit_cast<__int128>((long double)0);
#endif

// Same template instantiated with non-degenerate types does not warn.
unsigned ok = my_bit_cast<unsigned>(1);
} // namespace templates
