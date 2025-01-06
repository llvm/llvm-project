// RUN: %clang_cc1 -verify=expected,both -std=c++2a -fsyntax-only -triple x86_64-linux-gnu -fexperimental-new-constant-interpreter %s
// RUN: %clang_cc1 -verify=expected,both -std=c++2a -fsyntax-only -triple armv8 -fexperimental-new-constant-interpreter %s
// RUN: %clang_cc1 -verify=expected,both -std=c++2a -fsyntax-only -triple aarch64_be-linux-gnu -fexperimental-new-constant-interpreter %s
// RUN: %clang_cc1 -verify=expected,both -std=c++2a -fsyntax-only -fexperimental-new-constant-interpreter -triple powerpc64le-unknown-unknown -mabi=ieeelongdouble %s
// RUN: %clang_cc1 -verify=expected,both -std=c++2a -fsyntax-only -fexperimental-new-constant-interpreter -triple powerpc64-unknown-unknown -mabi=ieeelongdouble %s

#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
#  define LITTLE_END 1
#elif __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#  define LITTLE_END 0
#else
#  error "huh?"
#endif

typedef decltype(nullptr) nullptr_t;
typedef __INTPTR_TYPE__ intptr_t;
typedef unsigned __INT16_TYPE__ uint16_t;
typedef unsigned __INT32_TYPE__ uint32_t;
typedef unsigned __INT64_TYPE__ uint64_t;

static_assert(sizeof(int) == 4);
static_assert(sizeof(long long) == 8);

template <class To, class From>
constexpr To bit_cast(const From &from) {
  static_assert(sizeof(To) == sizeof(From));
  return __builtin_bit_cast(To, from);
}

template <class Intermediate, class Init>
constexpr bool check_round_trip(const Init &init) {
  return bit_cast<Init>(bit_cast<Intermediate>(init)) == init;
}

template <class Intermediate, class Init>
constexpr Init round_trip(const Init &init) {
  return bit_cast<Init>(bit_cast<Intermediate>(init));
}

namespace std {
enum byte : unsigned char {};
} // namespace std

template <int N, typename T = unsigned char, int Pad = 0>
struct bits {
  T : Pad;
  T bits : N;

  constexpr bool operator==(const T& rhs) const {
    return bits == rhs;
  }
};

template <int N, typename T, int P>
constexpr bool operator==(const struct bits<N, T, P>& lhs, const struct bits<N, T, P>& rhs) {
  return lhs.bits == rhs.bits;
}

template<int N>
struct bytes {
  using size_t = unsigned int;
  unsigned char d[N];

  constexpr unsigned char operator[](size_t index) {
    if (index < N)
      return d[index]; // expected-note {{read of uninitialized object}}
    return -1;
  }
};

namespace Sanity {
  /// This is just one byte, and we extract 2 bits from it.
  ///
  /// 3 is 0000'0011.
  /// For both LE and BE, the buffer will contain exactly that
  /// byte, unaltered and not reordered in any way. It contains all 8 bits.
  static_assert(__builtin_bit_cast(bits<2>, (unsigned char)3) == (LITTLE_END ? 3 : 0));

  /// Similarly, we have one full byte of data, with the two most-significant
  /// bits set:
  /// 192 is 1100'0000
  static_assert(__builtin_bit_cast(bits<2>, (unsigned char)192) == (LITTLE_END ? 0 : 3));


  /// Here we are instead bitcasting two 1-bits into a destination of 8 bits.
  /// On LE, we should pick the two least-significant bits. On BE, the opposite.
  /// NOTE: Can't verify this with gcc.
  constexpr auto B1 = bits<2>{3};
  static_assert(__builtin_bit_cast(unsigned char, B1) == (LITTLE_END ? 3 : 192));

  /// This should be 0000'0110.
  /// On LE, this should result in 6.
  /// On BE, 1100'0000 = 192.
  constexpr auto B2 = bits<3>{6};
  static_assert(__builtin_bit_cast(unsigned char, B2) == (LITTLE_END ? 6 : 192));

  constexpr auto B3 = bits<4>{6};
  static_assert(__builtin_bit_cast(unsigned char, B3) == (LITTLE_END ? 6 : 96));

  struct B {
    std::byte b0 : 4;
    std::byte b1 : 4;
  };

  /// We can properly decompose one byte (8 bit) int two 4-bit bitfields.
  constexpr struct { unsigned char b0; } T = {0xee};
  constexpr B MB = __builtin_bit_cast(B, T);
  static_assert(MB.b0 == 0xe);
  static_assert(MB.b1 == 0xe);
}

namespace BitFields {
  struct BitFields {
    unsigned a : 2;
    unsigned b : 30;
  };

  constexpr unsigned A = __builtin_bit_cast(unsigned, BitFields{3, 16});
  static_assert(A == (LITTLE_END ? 67 : 3221225488));

  struct S {
    unsigned a : 2;
    unsigned b : 28;
    unsigned c : 2;
  };

  constexpr S s = __builtin_bit_cast(S, 0xFFFFFFFF);
  static_assert(s.a == 3);
  static_assert(s.b == 268435455);
  static_assert(s.c == 3);

  void bitfield_indeterminate() {
    struct BF { unsigned char z : 2; };
    enum byte : unsigned char {};

    constexpr BF bf = {0x3};
    static_assert(bit_cast<bits<2>>(bf).bits == bf.z);
    static_assert(bit_cast<unsigned char>(bf));

    static_assert(__builtin_bit_cast(byte, bf)); // expected-error {{not an integral constant expression}} \
                                                 // expected-note {{indeterminate value can only initialize an object of type 'unsigned char' or 'std::byte'; 'byte' is invalid}}

    struct M {
      // expected-note@+1 {{subobject declared here}}
      unsigned char mem[sizeof(BF)];
    };
    // expected-error@+2 {{initialized by a constant expression}}
    // expected-note@+1 {{not initialized}}
    constexpr M m = bit_cast<M>(bf);

    constexpr auto f = []() constexpr {
      // bits<24, unsigned int, LITTLE_END ? 0 : 8> B = {0xc0ffee};
      constexpr struct { unsigned short b1; unsigned char b0;  } B = {0xc0ff, 0xee};
      return bit_cast<bytes<4>>(B);
    };

    static_assert(f()[0] + f()[1] + f()[2] == 0xc0 + 0xff + 0xee);
    {
      // expected-error@+2 {{initialized by a constant expression}}
      // expected-note@+1 {{in call to}}
      constexpr auto _bad = f()[3];
    }

    struct B {
      unsigned short s0 : 8;
      unsigned short s1 : 8;
      std::byte b0 : 4;
      std::byte b1 : 4;
      std::byte b2 : 4;
    };
    constexpr auto g = [f]() constexpr {
      return bit_cast<B>(f());
    };
    static_assert(g().s0 + g().s1 + g().b0 + g().b1 == 0xc0 + 0xff + 0xe + 0xe);
    {
      // expected-error@+2 {{initialized by a constant expression}}
      // expected-note@+1 {{read of uninitialized object is not allowed in a constant expression}}
      constexpr auto _bad = g().b2;
    }
  }
}

namespace BoolVectors {
  typedef bool bool32 __attribute__((ext_vector_type(32)));
  constexpr auto v = bit_cast<bool32>(0xa1c0ffee);
#if LITTLE_END
  static_assert(!v[0]);
  static_assert(v[1]);
  static_assert(v[2]);
  static_assert(v[3]);
  static_assert(!v[4]);
  static_assert(v[5]);
  static_assert(v[6]);
  static_assert(v[7]);

  static_assert(v[8]);
  static_assert(v[9]);
  static_assert(v[10]);
  static_assert(v[11]);
  static_assert(v[12]);
  static_assert(v[13]);
  static_assert(v[14]);
  static_assert(v[15]);

  static_assert(!v[16]);
  static_assert(!v[17]);
  static_assert(!v[18]);
  static_assert(!v[19]);
  static_assert(!v[20]);
  static_assert(!v[21]);
  static_assert(v[22]);
  static_assert(v[23]);

  static_assert(v[24]);
  static_assert(!v[25]);
  static_assert(!v[26]);
  static_assert(!v[27]);
  static_assert(!v[28]);
  static_assert(v[29]);
  static_assert(!v[30]);
  static_assert(v[31]);

#else
  static_assert(v[0]);
  static_assert(!v[1]);
  static_assert(v[2]);
  static_assert(!v[3]);
  static_assert(!v[4]);
  static_assert(!v[5]);
  static_assert(!v[6]);
  static_assert(v[7]);

  static_assert(v[8]);
  static_assert(v[9]);
  static_assert(!v[10]);
  static_assert(!v[11]);
  static_assert(!v[12]);
  static_assert(!v[13]);
  static_assert(!v[14]);
  static_assert(!v[15]);

  static_assert(v[16]);
  static_assert(v[17]);
  static_assert(v[18]);
  static_assert(v[19]);
  static_assert(v[20]);
  static_assert(v[21]);
  static_assert(v[22]);
  static_assert(v[23]);

  static_assert(v[24]);
  static_assert(v[25]);
  static_assert(v[26]);
  static_assert(!v[27]);
  static_assert(v[28]);
  static_assert(v[29]);
  static_assert(v[30]);
  static_assert(!v[31]);
#endif

  struct pad {
    unsigned short s;
    unsigned char c;
  };

  constexpr auto p = bit_cast<pad>(v);
  static_assert(p.s == (LITTLE_END ? 0xffee : 0xa1c0));
  static_assert(p.c == (LITTLE_END ? 0xc0 : 0xff));
}

namespace TwoShorts {
  struct B {
    unsigned short s0 : 8;
    unsigned short s1 : 8;
  };
  constexpr struct { unsigned short b1;} T = {0xc0ff};
  constexpr B MB = __builtin_bit_cast(B, T);
#if LITTLE_END
    static_assert(MB.s0 == 0xff);
    static_assert(MB.s1 == 0xc0);
#else
    static_assert(MB.s0 == 0xc0);
    static_assert(MB.s1 == 0xff);

#endif
}

typedef bool bool8 __attribute__((ext_vector_type(8)));
typedef bool bool9 __attribute__((ext_vector_type(9)));
typedef bool bool16 __attribute__((ext_vector_type(16)));
typedef bool bool17 __attribute__((ext_vector_type(17)));
typedef bool bool32 __attribute__((ext_vector_type(32)));
typedef bool bool128 __attribute__((ext_vector_type(128)));

static_assert(bit_cast<unsigned char>(bool8{1,0,1,0,1,0,1,0}) == (LITTLE_END ? 0x55 : 0xAA), "");
constexpr bool8 b8 = __builtin_bit_cast(bool8, 0x55); // both-error {{'__builtin_bit_cast' source type 'int' does not match destination type 'bool8' (vector of 8 'bool' values) (4 vs 1 bytes)}}
static_assert(check_round_trip<bool8>(static_cast<unsigned char>(0)), "");
static_assert(check_round_trip<bool8>(static_cast<unsigned char>(1)), "");
static_assert(check_round_trip<bool8>(static_cast<unsigned char>(0x55)), "");

static_assert(bit_cast<unsigned short>(bool16{1,1,1,1,1,0,0,0, 1,1,1,1,0,1,0,0}) == (LITTLE_END ? 0x2F1F : 0xF8F4), "");

static_assert(check_round_trip<bool16>(static_cast<short>(0xCAFE)), "");
static_assert(check_round_trip<bool32>(static_cast<int>(0xCAFEBABE)), "");

#ifdef __SIZEOF_INT128__
static_assert(check_round_trip<bool128>(static_cast<__int128_t>(0xCAFEBABE0C05FEFEULL)), "");
#endif

static_assert(bit_cast<bits<8, uint16_t, 7>, uint16_t>(0xcafe) == (LITTLE_END ? 0x95 : 0x7f));
static_assert(bit_cast<bits<4, uint16_t, 10>, uint16_t>(0xcafe) == (LITTLE_END ? 0x2 : 0xf));
static_assert(bit_cast<bits<4, uint32_t, 19>, uint32_t>(0xa1cafe) == (LITTLE_END ? 0x4 : 0x5));

struct S {
  // little endian:
  //    MSB .... .... LSB
  //        |y|   |x|
  //
  // big endian
  //    MSB .... .... LSB
  //        |x|   |y|

  unsigned char x : 4;
  unsigned char y : 4;

  constexpr bool operator==(S const &other) const {
    return x == other.x && y == other.y;
  }
};

constexpr S s{0xa, 0xb};
static_assert(bit_cast<bits<8>>(s) == (LITTLE_END ? 0xba : 0xab));
static_assert(bit_cast<bits<7>>(s) == (LITTLE_END
                                            ? 0xba & 0x7f
                                            : (0xab & 0xfe) >> 1));

static_assert(round_trip<bits<8>>(s) == s);

struct R {
  unsigned int r : 31;
  unsigned int : 0;
  unsigned int : 32;
  constexpr bool operator==(R const &other) const {
    return r == other.r;
  }
 };
using T = bits<31, signed long long>;
constexpr R r{0x4ac0ffee};
constexpr T t = bit_cast<T>(r);
static_assert(t == ((0xFFFFFFFF8 << 28) | 0x4ac0ffee)); // sign extension

static_assert(round_trip<T>(r) == r);
static_assert(round_trip<R>(t) == t);


/// The oversized bitfield is an error on Windows and not just a warning.
#if !defined(_WIN32)
struct U {
  // expected-warning@+1 {{exceeds the width of its type}}
  uint32_t trunc : 33;
  uint32_t u : 31;
  constexpr bool operator==(U const &other) const {
    return trunc == other.trunc && u == other.u;
  }
};
struct V {
  uint64_t notrunc : 32;
  uint64_t : 1;
  uint64_t v : 31;
  constexpr bool operator==(V const &other) const {
    return notrunc == other.notrunc && v == other.v;
  }
};

constexpr U u{static_cast<unsigned int>(~0), 0x4ac0ffee};
constexpr V v = bit_cast<V>(u);
static_assert(v.v == 0x4ac0ffee);

static_assert(round_trip<V>(u) == u);
static_assert(round_trip<U>(v) == v);

constexpr auto w = bit_cast<bits<12, unsigned long, 33>>(u);
static_assert(w == (LITTLE_END
                    ? 0x4ac0ffee & 0xFFF
                    : (0x4ac0ffee & (0xFFF << (31 - 12))) >> (31-12)
                  ));
#endif


namespace NestedStructures {
  struct J {
    struct {
      uint16_t  k : 12;
    } K;
    struct {
      uint16_t  l : 4;
    } L;
  };

  static_assert(sizeof(J) == 4);
  constexpr J j = bit_cast<J>(0x8c0ffee5);

  static_assert(j.K.k == (LITTLE_END ? 0xee5 : 0x8c0));
  static_assert(j.L.l == 0xf /* yay symmetry */);
  static_assert(bit_cast<bits<4, uint16_t, 16>>(j) == 0xf);
  struct N {
    bits<12, uint16_t> k;
    uint16_t : 16;
  };
  static_assert(bit_cast<N>(j).k == j.K.k);

  struct M {
    bits<4, uint16_t, 0> m[2];
    constexpr bool operator==(const M& rhs) const {
      return m[0] == rhs.m[0] && m[1] == rhs.m[1];
    };
  };
  #if LITTLE_END == 1
  constexpr uint16_t want[2] = {0x5, 0xf};
  #else
  constexpr uint16_t want[2] = {0x8000, 0xf000};
  #endif

  static_assert(bit_cast<M>(j) == bit_cast<M>(want));
}

namespace Enums {
  // ensure we're packed into the top 2 bits
  constexpr int pad = LITTLE_END ? 6 : 0;
  struct X
  {
    char : pad;
    enum class direction: char { left, right, up, down } direction : 2;
  };

  constexpr X x = { X::direction::down };
  static_assert(bit_cast<bits<2, signed char, pad>>(x) == -1);
  static_assert(bit_cast<bits<2, unsigned char, pad>>(x) == 3);
  static_assert(
    bit_cast<X>((unsigned char)0x40).direction == X::direction::right);
}

namespace IndeterminateBits {
  struct S {
    unsigned a : 13;
    unsigned   : 17;
    unsigned b : 2;
  };
  constexpr unsigned A = __builtin_bit_cast(unsigned, S{12, 3}); // expected-error {{must be initialized by a constant expression}} \
                                                                 // expected-note {{indeterminate value can only initialize an object of type 'unsigned char' or 'std::byte'; 'unsigned int' is invalid}}


  /// GCC refuses to compile this as soon as we access the indeterminate bits
  /// in the static_assert. MSVC accepts it.
  struct S2 {
    unsigned char a : 2;
  };
  constexpr unsigned char B = __builtin_bit_cast(unsigned char, S2{3});
  static_assert(B == (LITTLE_END ? 3 : 192));



  struct S3 {
    unsigned a : 13;
    unsigned   : 17;
    unsigned b : 2;
  };

  struct D {
    unsigned a;
  };
  constexpr D s = __builtin_bit_cast(D, S3{12, 3}); // expected-error {{must be initialized by a constant expression}} \
                                                    // expected-note {{indeterminate value can only initialize an object of type 'unsigned char' or 'std::byte'; 'unsigned int' is invalid}}

}
