// RUN: %clang_cc1 -verify -std=c++2a -fsyntax-only -triple x86_64-apple-macosx10.14.0 %s
// RUN: %clang_cc1 -verify -std=c++2a -fsyntax-only -triple x86_64-apple-macosx10.14.0 %s -fno-signed-char
// RUN: %clang_cc1 -verify -std=c++2a -fsyntax-only -triple aarch64_be-linux-gnu %s

// This is separate from constexpr-builtin-bit-cast.cpp because clangd17 seems to behave
// poorly around __BitInt(N) types, and this isolates that unfortunate behavior to one file
//
// hopefully a future clangd will not crash or lose track of its syntax highlighting, at which
// point the "bit_precise" namespace ought to be merged back into *bit-cast.cpp.

#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
#  define LITTLE_END 1
#elif __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#  define LITTLE_END 0
#else
#  error "huh?"
#endif

using uint8_t = unsigned char;

template <class To, class From>
constexpr To bit_cast(const From &from) {
  static_assert(sizeof(To) == sizeof(From));
  return __builtin_bit_cast(To, from);
}

namespace bit_precise {
// ok so it's a little bit of a lie to say we don't support _BitInt in any casts; we do in fact
// support casting _from_ a _BitInt(N), at least some of the time
static_assert(bit_cast<uint8_t, _BitInt(8)>(0xff) == 0xff);
template <int N> struct bytes { uint8_t b[N]; };
static_assert(bit_cast<bytes<2>, _BitInt(12)>(0xff).b[(LITTLE_END ? 0 : /* fixme */ 0)] == 0xff);
static_assert(bit_cast<bytes<4>, _BitInt(24)>(0xff).b[(LITTLE_END ? 0 : /* fixme */ 2)] == 0xff);

enum byte : unsigned char {}; // not std::byte

constexpr _BitInt(7) z = 0x7f;
constexpr auto bad_cast = __builtin_bit_cast(byte, z); // expected-error {{constant expression}}
// expected-note@-1 {{'bit_precise::byte' is invalid}}
// expected-note@-2 {{byte [0]}}

#if __clang_major__ > 17
// This is #ifdef'd off to stop clangd from crashing every time I open this file in my editor
// fixme? this crashes clang17 and before
constexpr auto unsupported_cast = __builtin_bit_cast(uint8_t, z); // expected-error {{constant expression}}
// expected-note@-1 {{subobject of type 'const uint8_t' (aka 'const unsigned char') is not initialized}}
#endif

// expected-note@+1 {{constexpr bit cast involving type '_BitInt(8)' is not yet supported}}
constexpr auto _n = __builtin_bit_cast(_BitInt(8), (uint8_t)0xff); // expected-error {{constant expression}}

// expected-note@+1 {{constexpr bit cast involving type '_BitInt(7)' is not yet supported}}
constexpr auto _m = __builtin_bit_cast(_BitInt(7), (uint8_t)0xff); // expected-error {{constant expression}}

// fixme: support _BitInt
// struct bitints {
//   _BitInt(2) x;
//   signed _BitInt(4) y;
// };
//
// constexpr auto bi = bit_cast<bitints, uint16_t>(0xff'ff);
// static_assert(bi.x == 0x3);
// static_assert(bi.y == -8);

// fixme?: the syntax highlighting here is a little off (`signed` and `constexpr` both lose their "keyword" coloring)
struct BF {
  _BitInt(2) x : 2;
  signed _BitInt(3) y : 2;
    // expected-warning@+1 {{exceeds the width of its type}}
  _BitInt(3) z : 4; // "oversized" bit field
};

// expected-note@+1 {{constexpr bit cast involving type '_BitInt(2)' is not yet supported}}
constexpr auto bf = __builtin_bit_cast(BF, (uint8_t)0xff); // expected-error {{must be initialized by a constant expression}}

// fixme: support _BitInt
// constexpr auto bf = bit_cast<BF, uint8_t>(0xff);
// static_assert(bf.x == 0x3);
// static_assert(bf.y == -4); // or +4 ?
// static_assert(bf.z == 0x7);

} // namespace bit_precise
