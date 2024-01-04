// RUN: %clang_cc1 -verify -std=c++2a -fsyntax-only -triple x86_64-apple-macosx10.14.0 %s
// RUN: %clang_cc1 -verify -std=c++2a -fsyntax-only -triple x86_64-apple-macosx10.14.0 %s -fno-signed-char
// RUN: %clang_cc1 -verify -std=c++2a -fsyntax-only -triple aarch64_be-linux-gnu %s

#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
#  define LITTLE_END 1
#elif __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#  define LITTLE_END 0
#else
#  error "huh?"
#endif

static_assert(sizeof(int) == 4);
static_assert(sizeof(long long) == 8);

using uint8_t = unsigned char;
using uint16_t = unsigned __INT16_TYPE__;
using uint32_t = unsigned __INT32_TYPE__;
using uint64_t = unsigned __INT64_TYPE__;

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

void test_int() {
  static_assert(check_round_trip<unsigned>((int)-1));
  static_assert(check_round_trip<unsigned>((int)0x12345678));
  static_assert(check_round_trip<unsigned>((int)0x87654321));
  static_assert(check_round_trip<unsigned>((int)0x0C05FEFE));
}

void test_array() {
  constexpr unsigned char input[] = {0xCA, 0xFE, 0xBA, 0xBE};
  constexpr unsigned expected = LITTLE_END ? 0xBEBAFECA : 0xCAFEBABE;
  static_assert(bit_cast<unsigned>(input) == expected);
}

void test_record() {
  struct int_splicer {
    unsigned x;
    unsigned y;

    constexpr bool operator==(const int_splicer &other) const {
      return other.x == x && other.y == y;
    }
  };

  constexpr int_splicer splice{0x0C05FEFE, 0xCAFEBABE};

  static_assert(bit_cast<unsigned long long>(splice) == (LITTLE_END
                                                             ? 0xCAFEBABE0C05FEFE
                                                             : 0x0C05FEFECAFEBABE));

  static_assert(bit_cast<int_splicer>(0xCAFEBABE0C05FEFE).x == (LITTLE_END
                                                                    ? 0x0C05FEFE
                                                                    : 0xCAFEBABE));

  static_assert(round_trip<unsigned long long>(splice) == splice);
  static_assert(round_trip<long long>(splice) == splice);

  struct base2 {
  };

  struct base3 {
    unsigned z;
  };

  struct bases : int_splicer, base2, base3 {
    unsigned doublez;
  };

  struct tuple4 {
    unsigned x, y, z, doublez;

    constexpr bool operator==(tuple4 const &other) const {
      return x == other.x && y == other.y &&
             z == other.z && doublez == other.doublez;
    }
  };
  constexpr bases b = {{1, 2}, {}, {3}, 4};
  constexpr tuple4 t4 = bit_cast<tuple4>(b);
  static_assert(t4 == tuple4{1, 2, 3, 4});
  static_assert(round_trip<tuple4>(b) == b);
}

void test_partially_initialized() {
  struct pad {
    signed char x;
    int y;
  };

  struct no_pad {
    signed char x;
    signed char p1, p2, p3;
    int y;
  };

  static_assert(sizeof(pad) == sizeof(no_pad));

  constexpr auto cast = [](const pad& from) constexpr {
    // expected-note@+6 2 {{bit_cast source expression (type 'const pad') does not produce a constant value for byte [1] (of {7..0}) which are required by target type 'no_pad' (subobject 'signed char')}}
    #ifdef __CHAR_UNSIGNED__
    // expected-note@+4 2 {{indeterminate value can only initialize an object of type 'unsigned char', 'char', or 'std::byte'; 'signed char' is invalid}}
    #else
    // expected-note@+2 2 {{indeterminate value can only initialize an object of type 'unsigned char' or 'std::byte'; 'signed char' is invalid}}
    #endif
    return __builtin_bit_cast(no_pad, from);
  };

  constexpr pad pir{4, 4};
  // expected-error@+2 {{constexpr variable 'piw' must be initialized by a constant expression}}
  // expected-note@+1 {{in call}}
  constexpr int piw = cast(pir).x;

  // expected-error@+2 {{constexpr variable 'bad' must be initialized by a constant expression}}
  // expected-note@+1 {{in call}}
  constexpr no_pad bad = cast(pir);

  constexpr pad fine = bit_cast<pad>(no_pad{1, 2, 3, 4, 5});
  static_assert(fine.x == 1 && fine.y == 5);
}

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

void test_bitfields() {
  {
    struct Q {
      // cf. CGBitFieldInfo
      // on a little-endian machine the bits "[count from] the
      // least-significant-bit."
      // so, by leaving a bit unused, we truncate the value's MSB.

      // however, on a big-endian machine we "imagine the bits
      // counting from the most-significant-bit", so we truncate
      // the LSB here.
      uint16_t q : 15;
    };
    constexpr unsigned char bytes[2] = {0xf3, 0xef};
    constexpr Q q = bit_cast<Q>(bytes);
    static_assert(q.q == (LITTLE_END ? 0x6ff3 : (0xf3ee >> 1)));
    static_assert(bit_cast<uint16_t>(bytes) == (LITTLE_END
                                                    ? 0xeff3
                                                    : 0xf3ef),
      "bit-field casting ought to match \"whole\"-field casting");

    // similarly, "skip 1 bit of padding" followed by "read 9 bits"
    // will truncate (shift out) either the LSB (little endian) or MSB (big endian)
    static_assert((0xf3ee >> 1) == 0x79f7);
    static_assert(0x01cf == (0xf3ef >> (16-9-1) & 0x1ff));
    static_assert(bit_cast<bits<9, uint16_t, 1>>(q) == (LITTLE_END
                                                              ? (0xeff3 >> 1) & 0x1ff
                                                              : (0xf3ef >> (16-9-1)) & 0x1ff));

    #if LITTLE_END == 0
    // expected-note@+5 {{bit [0]}}
    #else
    // expected-note@+3 {{bit [15]}}
    #endif
    // expected-error@+1 {{constant expression}}
    constexpr auto _i = __builtin_bit_cast(bits<15, uint16_t, 1>, q);
    // expected-note@-1 {{indeterminate}}
  }

  static_assert(round_trip<bits<8>, uint8_t>(0x8c) == 0x8c);
  static_assert(round_trip<bits<32, uint32_t>, uint32_t>(0x8c0f'fee5) == 0x8c0ffee5);

  #define MSG "endianness matters even with <=8-bit fields"
  static_assert(bit_cast<bits<8, uint16_t, 7>, uint16_t>(0xcafe) == (LITTLE_END
                                                                          ? 0x95
                                                                          : 0x7f), MSG);
  static_assert(bit_cast<bits<4, uint16_t, 10>, uint16_t>(0xcafe) == (LITTLE_END
                                                                          ? 0x2
                                                                          : 0xf), MSG);
  static_assert(bit_cast<bits<4, uint32_t, 19>, uint32_t>(0xa1cafe) == (LITTLE_END
                                                                          ? 0x4
                                                                          : 0x5), MSG);
  #undef MSG

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
  using T = bits<31, signed __INT64_TYPE__>;

  constexpr R r{0x4ac0ffee};
  constexpr T t = bit_cast<T>(r);
  static_assert(t == ((0xFFFFFFFF8 << 28) | 0x4ac0ffee)); // sign extension

  static_assert(round_trip<T>(r) == r);
  static_assert(round_trip<R>(t) == t);

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

  {
    #define MSG "a constexpr ought to produce padding bits from padding bits"
    static_assert(round_trip<V>(u) == u, MSG);
    static_assert(round_trip<U>(v) == v, MSG);

    constexpr auto w = bit_cast<bits<12, uint64_t, 33>>(u);
    static_assert(w == (LITTLE_END
                        ? 0x4ac0ffee & 0xFFF
                        : (0x4ac0ffee & (0xFFF << (31 - 12))) >> (31-12)
                      ), MSG);
    #undef MSG
  }

  // nested structures
  {
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

  // enums
  {
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
}

namespace std {
enum byte : unsigned char {};
} // namespace std

using uint8_t = unsigned char;

template<int N>
struct bytes {
  using size_t = unsigned int;
  unsigned char d[N];

  constexpr unsigned char &operator[](size_t index) {
    if (index < N)
      return d[index];
  }
};

void bitfield_indeterminate() {
  struct BF { unsigned char z : 2; };
  enum byte : unsigned char {};

  constexpr BF bf = {0x3};
  static_assert(bit_cast<bits<2>>(bf).bits == bf.z);

  // expected-error@+1 {{not an integral constant expression}}
  static_assert(bit_cast<unsigned char>(bf));
  /// FIXME the above doesn't get any helpful notes, but the below does
#if LITTLE_END == 1
  // expected-note@+6 {{bit [2-7]}}
#else
  // expected-note@+4 {{bit [0-5]}}
#endif
  // expected-note@+2 {{indeterminate}}
  // expected-error@+1 {{not an integral constant expression}}
  static_assert(__builtin_bit_cast(byte, bf));

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
    // expected-note@+1 {{read of uninitialized object is not allowed in a constant expression}}
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

void bitfield_unsupported() {
  // if a future standard requires more types to be permitted in the
  // declaration of a bit-field, then this test will hopefully indicate
  // that there's work to be done on __builtin_bit_cast.
  struct U {
    // expected-error@+1 {{bit-field 'f' has non-integral type}}
    bool f[8] : 8;
  };

  // this next bit is speculative: if the above _were_ a valid definition,
  // then the below might also be a reasonable interpretation of its
  // semantics, but the current implementation of __builtin_bit_cast will
  // fail

  // expected-note@+3 {{invalid declaration}} FIXME should we instead bail out in Sema?
  // expected-note@+2 {{declared here}}
  // expected-error@+1 {{initialized by a constant expression}}
  constexpr U u = __builtin_bit_cast(U, (char)0b1010'0101);
  static_assert(U.f[0] && U.f[2] && U.f[4] && U.f[8]);
  // expected-note@+2 {{not a constant expression}}
  // expected-error@+1 {{not an integral constant expression}}
  static_assert(__builtin_bit_cast(bits<8>, u) == 0xA5);
}

void array_members() {
  struct S {
    int ar[3];

    constexpr bool operator==(const S &rhs) {
      return ar[0] == rhs.ar[0] && ar[1] == rhs.ar[1] && ar[2] == rhs.ar[2];
    }
  };

  struct G {
    int a, b, c;

    constexpr bool operator==(const G &rhs) {
      return a == rhs.a && b == rhs.b && c == rhs.c;
    }
  };

  constexpr S s{{1, 2, 3}};
  constexpr G g = bit_cast<G>(s);
  static_assert(g.a == 1 && g.b == 2 && g.c == 3);

  static_assert(check_round_trip<G>(s));
  static_assert(check_round_trip<S>(g));
}

void bad_types() {
  union X {
    int x;
  };

  struct G {
    int g;
  };
  // expected-error@+2 {{constexpr variable 'g' must be initialized by a constant expression}}
  // expected-note@+1 {{bit_cast from a union type is not allowed in a constant expression}}
  constexpr G g = __builtin_bit_cast(G, X{0});
  // expected-error@+2 {{constexpr variable 'x' must be initialized by a constant expression}}
  // expected-note@+1 {{bit_cast to a union type is not allowed in a constant expression}}
  constexpr X x = __builtin_bit_cast(X, G{0});

  struct has_pointer {
    // expected-note@+1 2 {{invalid type 'int *' is a member of 'has_pointer'}}
    int *ptr;
  };

  // expected-error@+2 {{constexpr variable 'ptr' must be initialized by a constant expression}}
  // expected-note@+1 {{bit_cast from a pointer type is not allowed in a constant expression}}
  constexpr unsigned long ptr = __builtin_bit_cast(unsigned long, has_pointer{0});
  // expected-error@+2 {{constexpr variable 'hptr' must be initialized by a constant expression}}
  // expected-note@+1 {{bit_cast to a pointer type is not allowed in a constant expression}}
  constexpr has_pointer hptr =  __builtin_bit_cast(has_pointer, 0ul);
}

void backtrace() {
  struct A {
    // expected-note@+1 {{invalid type 'int *' is a member of 'A'}}
    int *ptr;
  };

  struct B {
    // expected-note@+1 {{invalid type 'A[10]' is a member of 'B'}}
    A as[10];
  };

  // expected-note@+1 {{invalid type 'B' is a base of 'C'}}
  struct C : B {
  };

  struct E {
    unsigned long ar[10];
  };

  // expected-error@+2 {{constexpr variable 'e' must be initialized by a constant expression}}
  // expected-note@+1 {{bit_cast from a pointer type is not allowed in a constant expression}}
  constexpr E e = __builtin_bit_cast(E, C{});
}

void test_array_fill() {
  constexpr unsigned char a[4] = {1, 2};
  constexpr unsigned int i = bit_cast<unsigned int>(a);
  static_assert(i == (LITTLE_END ? 0x00000201 : 0x01020000));
}

typedef decltype(nullptr) nullptr_t;

// expected-note@+7 {{byte [0-7]}}
#ifdef __CHAR_UNSIGNED__
// expected-note@+5 {{indeterminate value can only initialize an object of type 'unsigned char', 'char', or 'std::byte'; 'unsigned long' is invalid}}
#else
// expected-note@+3 {{indeterminate value can only initialize an object of type 'unsigned char' or 'std::byte'; 'unsigned long' is invalid}}
#endif
// expected-error@+1 {{constexpr variable 'test_from_nullptr' must be initialized by a constant expression}}
constexpr unsigned long test_from_nullptr = __builtin_bit_cast(unsigned long, nullptr);

constexpr int test_from_nullptr_pass = (__builtin_bit_cast(unsigned char[8], nullptr), 0);

constexpr int test_to_nullptr() {
  nullptr_t npt = __builtin_bit_cast(nullptr_t, 0ul);

  struct indet_mem {
    unsigned char data[sizeof(void *)];
  };
  indet_mem im = __builtin_bit_cast(indet_mem, nullptr);
  nullptr_t npt2 = __builtin_bit_cast(nullptr_t, im);

  return 0;
}

constexpr int ttn = test_to_nullptr();

// expected-warning@+2 {{returning reference to local temporary object}}
// expected-note@+1 {{temporary created here}}
constexpr const long &returns_local() { return 0L; }

// expected-error@+2 {{constexpr variable 'test_nullptr_bad' must be initialized by a constant expression}}
// expected-note@+1 {{read of temporary whose lifetime has ended}}
constexpr nullptr_t test_nullptr_bad = __builtin_bit_cast(nullptr_t, returns_local());

constexpr int test_indeterminate(bool read_indet) {
  struct pad {
    char a;
    int b;
  };

  struct no_pad {
    char a;
    unsigned char p1, p2, p3;
    int b;
  };

  pad p{1, 2};
  no_pad np = bit_cast<no_pad>(p);

  int tmp = np.a + np.b;

  unsigned char& indet_ref = np.p1;

  if (read_indet) {
    // expected-note@+1 {{read of uninitialized object is not allowed in a constant expression}}
    tmp = indet_ref;
  }

  indet_ref = 0;

  return 0;
}

constexpr int run_test_indeterminate = test_indeterminate(false);
// expected-error@+2 {{constexpr variable 'run_test_indeterminate2' must be initialized by a constant expression}}
// expected-note@+1 {{in call to 'test_indeterminate(true)'}}
constexpr int run_test_indeterminate2 = test_indeterminate(true);

struct ref_mem {
  const int &rm;
};

constexpr int global_int = 0;

// expected-error@+2 {{constexpr variable 'run_ref_mem' must be initialized by a constant expression}}
// expected-note@+1 {{bit_cast from a type with a reference member is not allowed in a constant expression}}
constexpr unsigned long run_ref_mem = __builtin_bit_cast(
    unsigned long, ref_mem{global_int});

union u {
  int im;
};

// expected-error@+2 {{constexpr variable 'run_u' must be initialized by a constant expression}}
// expected-note@+1 {{bit_cast from a union type is not allowed in a constant expression}}
constexpr int run_u = __builtin_bit_cast(int, u{32});

struct vol_mem {
  volatile int x;
};

// expected-error@+2 {{constexpr variable 'run_vol_mem' must be initialized by a constant expression}}
// expected-note@+1 {{non-literal type 'vol_mem' cannot be used in a constant expression}}
constexpr int run_vol_mem = __builtin_bit_cast(int, vol_mem{43});

struct mem_ptr {
  int vol_mem::*x; // expected-note{{invalid type 'int vol_mem::*' is a member of 'mem_ptr'}}
};
// expected-error@+2 {{constexpr variable 'run_mem_ptr' must be initialized by a constant expression}}
// expected-note@+1 {{bit_cast from a member pointer type is not allowed in a constant expression}}
constexpr int run_mem_ptr = __builtin_bit_cast(unsigned long, mem_ptr{nullptr});

struct A { char c; /* char padding : 8; */ short s; };
struct B { unsigned char x[4]; };

constexpr B one() {
  A a = {1, 2};
  return bit_cast<B>(a);
}
constexpr char good_one = one().x[0] + one().x[2] + one().x[3];
// expected-error@+2 {{constexpr variable 'bad_one' must be initialized by a constant expression}}
// expected-note@+1 {{read of uninitialized object is not allowed in a constant expression}}
constexpr char bad_one = one().x[1];

constexpr A two() {
  B b = one(); // b.x[1] is indeterminate.
  b.x[0] = 'a';
  b.x[2] = 1;
  b.x[3] = 2;
  return bit_cast<A>(b);
}
constexpr short good_two = two().c + two().s;

enum my_byte : unsigned char {};

struct pad {
  char a;
  int b;
};

constexpr int ok_byte = (__builtin_bit_cast(std::byte[8], pad{1, 2}), 0);
constexpr int ok_uchar = (__builtin_bit_cast(unsigned char[8], pad{1, 2}), 0);

// expected-note@+7 {{bit_cast source expression (type 'pad') does not produce a constant value for byte [1] (of {7..0}) which are required by target type 'my_byte[8]' (subobject 'my_byte')}}
#ifdef __CHAR_UNSIGNED__
// expected-note@+5 {{indeterminate value can only initialize an object of type 'unsigned char', 'char', or 'std::byte'; 'my_byte' is invalid}}
#else
// expected-note@+3 {{indeterminate value can only initialize an object of type 'unsigned char' or 'std::byte'; 'my_byte' is invalid}}
#endif
// expected-error@+1 {{constexpr variable 'bad_my_byte' must be initialized by a constant expression}}
constexpr int bad_my_byte = (__builtin_bit_cast(my_byte[8], pad{1, 2}), 0);
#ifndef __CHAR_UNSIGNED__
// expected-note@+4 {{bit_cast source expression (type 'pad') does not produce a constant value for byte [1] (of {7..0}) which are required by target type 'char[8]' (subobject 'char')}}
// expected-note@+3 {{indeterminate value can only initialize an object of type 'unsigned char' or 'std::byte'; 'char' is invalid}}
// expected-error@+2 {{constexpr variable 'bad_char' must be initialized by a constant expression}}
#endif
constexpr int bad_char =  (__builtin_bit_cast(char[8], pad{1, 2}), 0);

struct pad_buffer { unsigned char data[sizeof(pad)]; };
constexpr bool test_pad_buffer() {
  pad x = {1, 2};
  pad_buffer y = __builtin_bit_cast(pad_buffer, x);
  pad z = __builtin_bit_cast(pad, y);
  return x.a == z.a && x.b == z.b;
}
static_assert(test_pad_buffer());

constexpr unsigned char identity1a = 42;
constexpr unsigned char identity1b = __builtin_bit_cast(unsigned char, identity1a);
static_assert(identity1b == 42);

struct IdentityInStruct {
  unsigned char n;
};
constexpr IdentityInStruct identity2a = {42};
constexpr unsigned char identity2b = __builtin_bit_cast(unsigned char, identity2a.n);

union IdentityInUnion {
  unsigned char n;
};
constexpr IdentityInUnion identity3a = {42};
constexpr unsigned char identity3b = __builtin_bit_cast(unsigned char, identity3a.n);

namespace test_bool {

// expected-note@+1 {{cannot be represented in type 'bool'}}
constexpr bool test_bad_bool = __builtin_bit_cast(bool, 'A'); // expected-error {{must be initialized by a constant expression}}

static_assert(round_trip<signed char>(true));
static_assert(round_trip<unsigned char>(true));
static_assert(round_trip<bool>(false) == false);

static_assert(static_cast<uint8_t>(false) == 0x0);
static_assert(bit_cast<uint8_t>(false) == 0x0);
static_assert(static_cast<uint8_t>(true) == 0x1);
static_assert(bit_cast<uint8_t>(true) == 0x1);

static_assert(round_trip<bool, uint8_t>(0x01) == 0x1);
static_assert(round_trip<bool, uint8_t>(0x00) == 0x0);
// expected-note@+2 {{cannot be represented in type 'bool'}}
// expected-error@+1 {{constant expression}}
constexpr auto test_bad_bool2 = __builtin_bit_cast(bool, (uint8_t)0x02);

#if LITTLE_END == 1
constexpr auto okbits = bit_cast<bits<1>>(true);
#else
constexpr auto okbits = bit_cast<bits<1, uint8_t, 7>>(true);
#endif
static_assert(okbits == 0x1);
// expected-note@+3 {{bit [1-7]}}
// expected-note@+2 {{or 'std::byte'; 'bool' is invalid}}
// expected-error@+1 {{constant expression}}
constexpr auto _weird_bool = __builtin_bit_cast(bool, okbits);

// these don't work because we're trying to read the whole 8 bits to ensure
// the value is representable, as above
// static_assert(round_trip<bool, bits<1>>({0x1}) == 0x1);
// static_assert(round_trip<bool, bits<1>>({0x0}) == 0x0);

// these work because we're only reading 1 bit of "bool" to ensure
// "representability"
static_assert(round_trip<bits<1, bool>, bits<1>>({0x1}) == 0x1);
static_assert(round_trip<bits<1, bool>, bits<1>>({0x0}) == 0x0);

template <const int P, class B = bool>
constexpr bool extract_bit(unsigned char v) {
  return static_cast<bool>(bit_cast<bits<1, B, P>>(v).bits);
}
// 0xA5 is a palindrome, so endianness doesn't matter
// (counting LSB->MSB is the same as MSB->LSB)
static_assert(extract_bit<0>(0xA5) == 0x1);
static_assert(extract_bit<2>(0xA5) == 0x1);
static_assert(extract_bit<5>(0xA5) == 0x1);
static_assert(extract_bit<7>(0xA5) == 0x1);

static_assert(extract_bit<1>(0xA5) == 0x0);
static_assert(extract_bit<3>(0xA5) == 0x0);
static_assert(extract_bit<4>(0xA5) == 0x0);
static_assert(extract_bit<6>(0xA5) == 0x0);

enum byte : unsigned char {}; // not std::byte or unsigned char

static_assert(extract_bit<5, byte>('\xa5') == 0x1);

struct pad {
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  bool : 5; // push field down to the LSB
#endif
  bool b : 3;
};

static_assert(bit_cast<pad, uint8_t>(0b001).b == true);
static_assert(bit_cast<pad, uint8_t>(0b000).b == false);

// expected-note@+1 {{cannot be represented in type 'bool'}}
constexpr auto _bad_bool3 = __builtin_bit_cast(pad, (uint8_t)0b110); // expected-error {{must be initialized by a constant expression}}

struct S {
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  byte : 7;
#endif
  byte z : 1;
};

constexpr auto s = bit_cast<S>(pad{1});
static_assert(s.z == 0x1);

// expected-note@+3 {{bit [1-2]}}
// expected-note@+2 {{or 'std::byte'; 'bool' is invalid}}
// expected-error@+1 {{constant expression}}
constexpr auto _bad_bool4 = __builtin_bit_cast(pad, s);


// `bool` includes padding bits, but *which* single bit stores the
// value is under-specified. These tests not-so-secretly assert that
// it's in fact the LSB that the compiler "sees" as the value.
struct pack {
  bool a : 1;
  bool b : 1;

  // 1 bit of value, 5 bits of padding
  bool c : 6;
};

constexpr auto packed = bit_cast<pack, uint8_t>(LITTLE_END ? 0x07 : 0xc1);
static_assert(packed.a && packed.b && packed.c);

static_assert(bit_cast<bits<2, uint8_t, 0>>(packed) == 0x3);
static_assert(bit_cast<bits<1, uint8_t, LITTLE_END ? 2 : 7>>(packed) == 0x1);

} // namespace test_bool

namespace test_long_double {
#ifdef __x86_64
// expected-note@+2 {{byte [10-15]}}
// expected-note@+1 {{or 'std::byte'; '__int128' is invalid}}
constexpr __int128_t test_cast_to_int128 = __builtin_bit_cast(__int128_t, (long double)0); // expected-error{{must be initialized by a constant expression}}

constexpr long double ld = 3.1425926539;

using bytes = bytes<16>;

static_assert(check_round_trip<bytes>(ld));

static_assert(check_round_trip<long double>(10.0L));

constexpr bool f(bool read_uninit) {
  bytes b = bit_cast<bytes>(ld);
  unsigned char ld_bytes[10] = {
    0x0,  0x48, 0x9f, 0x49, 0xf0,
    0x3c, 0x20, 0xc9, 0x0,  0x40,
  };

  for (int i = 0; i != 10; ++i)
    if (ld_bytes[i] != b[i])
      return false;

  if (read_uninit && b[10]) // expected-note{{read of uninitialized object is not allowed in a constant expression}}
    return false;

  return true;
}

static_assert(f(/*read_uninit=*/false));
static_assert(f(/*read_uninit=*/true)); // expected-error{{static assertion expression is not an integral constant expression}} expected-note{{in call to 'f(true)'}}

constexpr bytes ld539 = {
  0x0, 0x0,  0x0,  0x0,
  0x0, 0x0,  0xc0, 0x86,
  0x8, 0x40, 0x0,  0x0,
  0x0, 0x0,  0x0,  0x0,
};

constexpr long double fivehundredandthirtynine = 539.0;

static_assert(bit_cast<long double>(ld539) == fivehundredandthirtynine);

#else
static_assert(round_trip<__int128_t>(34.0L));
#endif
} // namespace test_long_double

namespace test_vector {

typedef unsigned uint2 __attribute__((vector_size(2 * sizeof(unsigned))));
typedef char byte8 __attribute__((vector_size(sizeof(unsigned long long))));

constexpr uint2 test_vector = { 0x0C05FEFE, 0xCAFEBABE };

static_assert(bit_cast<unsigned long long>(test_vector) == (LITTLE_END
                                                                ? 0xCAFEBABE0C05FEFE
                                                                : 0x0C05FEFECAFEBABE));

static_assert(check_round_trip<uint2>(0xCAFEBABE0C05FEFEULL));
static_assert(check_round_trip<byte8>(0xCAFEBABE0C05FEFEULL));

typedef bool bool8 __attribute__((ext_vector_type(8)));
typedef bool bool9 __attribute__((ext_vector_type(9)));
typedef bool bool16 __attribute__((ext_vector_type(16)));
typedef bool bool17 __attribute__((ext_vector_type(17)));
typedef bool bool32 __attribute__((ext_vector_type(32)));
typedef bool bool128 __attribute__((ext_vector_type(128)));

static_assert(bit_cast<unsigned char>(bool8{1,0,1,0,1,0,1,0}) == (LITTLE_END ? 0x55 : 0xAA));
static_assert(round_trip<bool8>('\x00') == 0);
static_assert(round_trip<bool8>('\x01') == 0x1);
static_assert(round_trip<bool8>('\x55') == 0x55);

static_assert(bit_cast<unsigned short>(bool16{1,1,1,1,1,0,0,0, 1,1,1,1,0,1,0,0}) == (LITTLE_END ? 0x2F1F : 0xF8F4));

static_assert(check_round_trip<bool16>(static_cast<short>(0xCAFE)));
static_assert(check_round_trip<bool32>(static_cast<int>(0xCAFEBABE)));
static_assert(check_round_trip<bool128>(static_cast<__int128_t>(0xCAFEBABE0C05FEFEULL)));

// expected-error@+2 {{constexpr variable 'bad_bool9_to_short' must be initialized by a constant expression}}
// expected-note@+1 {{bit_cast involving type 'bool __attribute__((ext_vector_type(9)))' (vector of 9 'bool' values) is not allowed in a constant expression; element size 1 * element count 9 is not a multiple of the byte size 8}}
constexpr unsigned short bad_bool9_to_short = __builtin_bit_cast(unsigned short, bool9{1,1,0,1,0,1,0,1,0});
// expected-error@+2 {{constexpr variable 'bad_short_to_bool9' must be initialized by a constant expression}}
// expected-note@+1 {{bit_cast involving type 'bool __attribute__((ext_vector_type(9)))' (vector of 9 'bool' values) is not allowed in a constant expression; element size 1 * element count 9 is not a multiple of the byte size 8}}
constexpr bool9 bad_short_to_bool9 = __builtin_bit_cast(bool9, static_cast<unsigned short>(0));
// expected-error@+2 {{constexpr variable 'bad_int_to_bool17' must be initialized by a constant expression}}
// expected-note@+1 {{bit_cast involving type 'bool __attribute__((ext_vector_type(17)))' (vector of 17 'bool' values) is not allowed in a constant expression; element size 1 * element count 17 is not a multiple of the byte size 8}}
constexpr bool17 bad_int_to_bool17 = __builtin_bit_cast(bool17, 0x0001CAFEU);

struct pad {
  unsigned short s;
  unsigned char c;
};
constexpr auto p = bit_cast<pad>(bit_cast<bool32>(0xa1c0ffee));
static_assert(p.s == (LITTLE_END ? 0xffee : 0xa1c0));
static_assert(p.c == (LITTLE_END ? 0xc0 : 0xff));

#if LITTLE_END == 1
// expected-note@+5 {{for byte [3]}}
#else
// expected-note@+3 {{for byte [0]}}
#endif
// expected-note@+1 {{indeterminate value}}
constexpr auto _bad_p = __builtin_bit_cast(bool32, p); // expected-error {{initialized by a constant expression}}


} // namespace test_vector
