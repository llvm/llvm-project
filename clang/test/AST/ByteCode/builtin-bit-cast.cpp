// RUN: %clang_cc1 -verify=ref,both -std=c++2a -fsyntax-only %s
// RUN: %clang_cc1 -verify=ref,both -std=c++2a -fsyntax-only -triple aarch64_be-linux-gnu %s
// RUN: %clang_cc1 -verify=ref,both -std=c++2a -fsyntax-only -triple powerpc64le-unknown-unknown -mabi=ieeelongdouble %s
// RUN: %clang_cc1 -verify=ref,both -std=c++2a -fsyntax-only -triple powerpc64-unknown-unknown -mabi=ieeelongdouble %s

// RUN: %clang_cc1 -verify=expected,both -std=c++2a -fsyntax-only -fexperimental-new-constant-interpreter %s
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

static_assert(sizeof(int) == 4);
static_assert(sizeof(long long) == 8);


constexpr bool test_bad_bool = __builtin_bit_cast(bool, (char)0xff); // both-error {{must be initialized by a constant expression}} \
                                                                     // both-note {{value 255 cannot be represented in type 'bool'}}

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


namespace Discarding {
  struct S { int a; };
  constexpr int f = (__builtin_bit_cast(int, 2), 0);
  constexpr int f2 = (__builtin_bit_cast(S, 2), 0);
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

#ifdef __SIZEOF_INT128__
static_assert(check_round_trip<__int128_t>((__int128_t)34));
static_assert(check_round_trip<__int128_t>((__int128_t)-34));

constexpr unsigned char OneBit[] = {
  0x1, 0x0,  0x0,  0x0,
  0x0, 0x0,  0x0,  0x0,
  0x0, 0x0,  0x0,  0x0,
  0x0, 0x0,  0x0,  0x0,
};
constexpr __int128_t One = 1;
constexpr __int128_t Expected = One << 120;
static_assert(__builtin_bit_cast(__int128_t, OneBit) == (LITTLE_END ? 1 : Expected));

#endif

static_assert(check_round_trip<double>(17.0));


namespace simple {
  constexpr int A = __builtin_bit_cast(int, 10);
  static_assert(A == 10);

  static_assert(__builtin_bit_cast(unsigned, 1.0F) == 1065353216);

  struct Bytes {
    char a, b, c, d;
  };
  constexpr unsigned B = __builtin_bit_cast(unsigned, Bytes{10, 12, 13, 14});
  static_assert(B == (LITTLE_END ? 235736074 : 168561934));


  constexpr unsigned C = __builtin_bit_cast(unsigned, (_BitInt(32))12);
  static_assert(C == 12);

  struct BitInts {
    _BitInt(16) a;
    _BitInt(16) b;
  };
  constexpr unsigned D = __builtin_bit_cast(unsigned, BitInts{12, 13});
  static_assert(D == (LITTLE_END ? 851980 : 786445));



  static_assert(__builtin_bit_cast(char, true) == 1);

  static_assert(check_round_trip<unsigned>((int)-1));
  static_assert(check_round_trip<unsigned>((int)0x12345678));
  static_assert(check_round_trip<unsigned>((int)0x87654321));
  static_assert(check_round_trip<unsigned>((int)0x0C05FEFE));
  static_assert(round_trip<float>((int)0x0C05FEFE));

  static_assert(__builtin_bit_cast(intptr_t, nullptr) == 0); // both-error {{not an integral constant expression}} \
                                                             // both-note {{indeterminate value can only initialize an object}}

  constexpr int test_from_nullptr_pass = (__builtin_bit_cast(unsigned char[sizeof(nullptr)], nullptr), 0);
  constexpr unsigned char NPData[sizeof(nullptr)] = {1,2,3,4};
  constexpr nullptr_t NP = __builtin_bit_cast(nullptr_t, NPData);
  static_assert(NP == nullptr);
}

namespace Fail {
  constexpr int a = 1/0; // both-error {{must be initialized by a constant expression}} \
                         // both-note {{division by zero}} \
                         // both-note {{declared here}}
  constexpr int b = __builtin_bit_cast(int, a); // both-error {{must be initialized by a constant expression}} \
                                                // both-note {{initializer of 'a' is not a constant expression}}
}

namespace ToPtr {
  struct S {
    const int *p = nullptr;
  };
  struct P {
    const int *p; // both-note {{invalid type 'const int *' is a member of 'ToPtr::P'}}
  };
  constexpr P p = __builtin_bit_cast(P, S{}); // both-error {{must be initialized by a constant expression}} \
                                              // both-note {{bit_cast to a pointer type is not allowed in a constant expression}}
}

namespace Invalid {
  struct S {
    int a;
  };
  constexpr S s = S{1/0}; // both-error {{must be initialized by a constant expression}} \
                          // both-note {{division by zero}} \
                          // both-note {{declared here}}
  constexpr S s2 = __builtin_bit_cast(S, s); // both-error {{must be initialized by a constant expression}} \
                                             // both-note {{initializer of 's' is not a constant expression}}
}

namespace NullPtr {
  constexpr nullptr_t N = __builtin_bit_cast(nullptr_t, (intptr_t)1u);
  static_assert(N == nullptr);
  static_assert(__builtin_bit_cast(nullptr_t, (_BitInt(sizeof(void*) * 8))12) == __builtin_bit_cast(nullptr_t, (unsigned _BitInt(sizeof(void*) * 8))0));
  static_assert(__builtin_bit_cast(nullptr_t, nullptr) == nullptr);
}

namespace bitint {
  constexpr _BitInt(sizeof(int) * 8) BI = ~0;
  constexpr unsigned int I = __builtin_bit_cast(unsigned int, BI);
  static_assert(I == ~0u, "");

  constexpr _BitInt(sizeof(int) * 8) IB = __builtin_bit_cast(_BitInt(sizeof(int) * 8), I); // ref-error {{must be initialized by a constant expression}} \
                                                                                           // ref-note {{constexpr bit cast involving type '_BitInt(32)' is not yet supported}} \
                                                                                           // ref-note {{declared here}}
  static_assert(IB == ~0u, ""); // ref-error {{not an integral constant expression}} \
                                // ref-note {{initializer of 'IB' is not a constant expression}}
}

namespace Classes {
  class A {
  public:
    char a[2];
  };
  class B : public A {
  public:
    char b[2];
  };
  static_assert(__builtin_bit_cast(int, B{{0,  0},{0,  0}}) == 0);
  static_assert(__builtin_bit_cast(int, B{{13, 0},{0,  0}}) == (LITTLE_END ? 13 : 218103808));
  static_assert(__builtin_bit_cast(int, B{{13, 7},{12, 20}}) == (LITTLE_END ? 336332557 : 218565652));

  class Ref {
  public:
    const int &a;
    constexpr Ref(const int &a) : a(a) {}
  };
  constexpr int I = 12;

  typedef __INTPTR_TYPE__ intptr_t;
  static_assert(__builtin_bit_cast(intptr_t, Ref{I}) == 0); // both-error {{not an integral constant expression}} \
                                                            // both-note {{bit_cast from a type with a reference member is not allowed in a constant expression}}

  class C : public A {
    public:
    constexpr C() : A{1,2} {}
    virtual constexpr int get() {
      return 4;
    }
  };
  static_assert(__builtin_bit_cast(_BitInt(sizeof(C) * 8), C()) == 0); // both-error {{source type must be trivially copyable}}


  class D : virtual A {};
  static_assert(__builtin_bit_cast(_BitInt(sizeof(D) * 8), D()) == 0); // both-error {{source type must be trivially copyable}}

  class F {
  public:
    char f[2];
  };

  class E : public A, public F {
  public:
    constexpr E() : A{1,2}, F{3,4}, e{5,6,7,8} {}
    char e[4];
  };
  static_assert(__builtin_bit_cast(long long, E()) == (LITTLE_END ? 578437695752307201 : 72623859790382856));
}

struct int_splicer {
  unsigned x;
  unsigned y;

  constexpr int_splicer() : x(1), y(2) {}
  constexpr int_splicer(unsigned x, unsigned y) : x(x), y(y) {}

  constexpr bool operator==(const int_splicer &other) const {
    return other.x == x && other.y == y;
  }
};

constexpr int_splicer splice(0x0C05FEFE, 0xCAFEBABE);

#if 1
static_assert(bit_cast<unsigned long long>(splice) == (LITTLE_END
                                                           ? 0xCAFEBABE0C05FEFE
                                                           : 0x0C05FEFECAFEBABE));

constexpr int_splicer IS = bit_cast<int_splicer>(0xCAFEBABE0C05FEFE);
static_assert(bit_cast<int_splicer>(0xCAFEBABE0C05FEFE).x == (LITTLE_END
                                                                  ? 0x0C05FEFE
                                                                  : 0xCAFEBABE));

static_assert(check_round_trip<unsigned long long>(splice));
static_assert(check_round_trip<long long>(splice));
#endif


namespace Overread {
  /// This used to crash becaus we were reading all elements of the
  /// source array even though we should only be reading 1.
  constexpr int a[] = {2,3, 4, 5};
  constexpr int b = __builtin_bit_cast(int, *(a + 1));
  static_assert(b == 3);

  struct S {
    int a;
  };
  constexpr S ss[] = {{1},{2}};
  constexpr int c = __builtin_bit_cast(int, *(ss + 1));
  static_assert(c == 2);
}


/// ---------------------------------------------------------------------------
/// From here on, it's things copied from test/SemaCXX/constexpr-builtin-bit.cast.cpp

void test_int() {
  static_assert(round_trip<unsigned>((int)-1));
  static_assert(round_trip<unsigned>((int)0x12345678));
  static_assert(round_trip<unsigned>((int)0x87654321));
  static_assert(round_trip<unsigned>((int)0x0C05FEFE));
}

void test_array() {
  constexpr unsigned char input[] = {0xCA, 0xFE, 0xBA, 0xBE};
  constexpr unsigned expected = LITTLE_END ? 0xBEBAFECA : 0xCAFEBABE;
  static_assert(bit_cast<unsigned>(input) == expected);

  /// Same things but with a composite array.
  struct US { unsigned char I; };
  constexpr US input2[] = {{0xCA}, {0xFE}, {0xBA}, {0xBE}};
  static_assert(bit_cast<unsigned>(input2) == expected);
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

  static_assert(check_round_trip<unsigned long long>(splice));
  static_assert(check_round_trip<long long>(splice));

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

    bool operator==(tuple4 const &other) const = default;
    constexpr bool operator==(bases const &other) const {
      return x == other.x && y == other.y &&
             z == other.z && doublez == other.doublez;
    }
  };
  constexpr bases b = {{1, 2}, {}, {3}, 4};
  constexpr tuple4 t4 = bit_cast<tuple4>(b);
  static_assert(t4 == tuple4{1, 2, 3, 4});
  static_assert(check_round_trip<tuple4>(b));

  constexpr auto b2 = bit_cast<bases>(t4);
  static_assert(t4 == b2);
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

#if 0
  constexpr pad pir{4, 4};
  constexpr int piw = bit_cast<no_pad>(pir).x; // both-error {{constexpr variable 'piw' must be initialized by a constant expression}} \
                                               // both-note {{in call to 'bit_cast<no_pad, pad>(pir)'}}


  constexpr no_pad bad = bit_cast<no_pad>(pir); // both-error {{constexpr variable 'bad' must be initialized by a constant expression}} \
                                                // both-note {{in call to 'bit_cast<no_pad, pad>(pir)'}}
  // constexpr pad fine = bit_cast<pad>(no_pad{1, 2, 3, 4, 5});
  // static_assert(fine.x == 1 && fine.y == 5);
#endif
}


void bad_types() {
  union X {
    int x;
  };
  static_assert(__builtin_bit_cast(int, X{0}) == 0); // both-error {{not an integral constant expression}} \
                                                     // both-note {{bit_cast from a union type is not allowed in a constant expression}}

  struct G {
    int g;
  };
  // both-error@+2 {{constexpr variable 'g' must be initialized by a constant expression}}
  // both-note@+1 {{bit_cast from a union type is not allowed in a constant expression}}
  constexpr G g = __builtin_bit_cast(G, X{0});
  // both-error@+2 {{constexpr variable 'x' must be initialized by a constant expression}}
  // both-note@+1 {{bit_cast to a union type is not allowed in a constant expression}}
  constexpr X x = __builtin_bit_cast(X, G{0});

  struct has_pointer {
    int *ptr; // both-note 2{{invalid type 'int *' is a member of 'has_pointer'}}
  };

  constexpr intptr_t ptr = __builtin_bit_cast(intptr_t, has_pointer{0}); // both-error {{constexpr variable 'ptr' must be initialized by a constant expression}} \
                                                                         // both-note {{bit_cast from a pointer type is not allowed in a constant expression}}

  // both-error@+2 {{constexpr variable 'hptr' must be initialized by a constant expression}}
  // both-note@+1 {{bit_cast to a pointer type is not allowed in a constant expression}}
  constexpr has_pointer hptr =  __builtin_bit_cast(has_pointer, (intptr_t)0);
}

void test_array_fill() {
  constexpr unsigned char a[4] = {1, 2};
  constexpr unsigned int i = bit_cast<unsigned int>(a);
  static_assert(i == (LITTLE_END ? 0x00000201 : 0x01020000));
}

struct vol_mem {
  volatile int x;
};

// both-error@+2 {{constexpr variable 'run_vol_mem' must be initialized by a constant expression}}
// both-note@+1 {{non-literal type 'vol_mem' cannot be used in a constant expression}}
constexpr int run_vol_mem = __builtin_bit_cast(int, vol_mem{43});

struct mem_ptr {
  int vol_mem::*x; // both-note{{invalid type 'int vol_mem::*' is a member of 'mem_ptr'}}
};

// both-error@+2 {{constexpr variable 'run_mem_ptr' must be initialized by a constant expression}}
// both-note@+1 {{bit_cast from a member pointer type is not allowed in a constant expression}}
constexpr _BitInt(sizeof(mem_ptr) * 8) run_mem_ptr = __builtin_bit_cast(_BitInt(sizeof(mem_ptr) * 8), mem_ptr{nullptr});

constexpr int global_int = 0;

struct ref_mem {
  const int &rm;
};
// both-error@+2 {{constexpr variable 'run_ref_mem' must be initialized by a constant expression}}
// both-note@+1 {{bit_cast from a type with a reference member is not allowed in a constant expression}}
constexpr intptr_t run_ref_mem = __builtin_bit_cast(intptr_t, ref_mem{global_int});

namespace test_vector {

typedef unsigned uint2 __attribute__((vector_size(2 * sizeof(unsigned))));
typedef char byte8 __attribute__((vector_size(sizeof(unsigned long long))));

constexpr uint2 test_vector = { 0x0C05FEFE, 0xCAFEBABE };

static_assert(bit_cast<unsigned long long>(test_vector) == (LITTLE_END
                                                                ? 0xCAFEBABE0C05FEFE
                                                                : 0x0C05FEFECAFEBABE), "");
static_assert(check_round_trip<uint2>(0xCAFEBABE0C05FEFEULL), "");
static_assert(check_round_trip<byte8>(0xCAFEBABE0C05FEFEULL), "");

#if 0
// expected-error@+2 {{constexpr variable 'bad_bool9_to_short' must be initialized by a constant expression}}
// expected-note@+1 {{bit_cast involving type 'bool __attribute__((ext_vector_type(9)))' (vector of 9 'bool' values) is not allowed in a constant expression; element size 1 * element count 9 is not a multiple of the byte size 8}}
constexpr unsigned short bad_bool9_to_short = __builtin_bit_cast(unsigned short, bool9{1,1,0,1,0,1,0,1,0});
// expected-error@+2 {{constexpr variable 'bad_short_to_bool9' must be initialized by a constant expression}}
// expected-note@+1 {{bit_cast involving type 'bool __attribute__((ext_vector_type(9)))' (vector of 9 'bool' values) is not allowed in a constant expression; element size 1 * element count 9 is not a multiple of the byte size 8}}
constexpr bool9 bad_short_to_bool9 = __builtin_bit_cast(bool9, static_cast<unsigned short>(0));
// expected-error@+2 {{constexpr variable 'bad_int_to_bool17' must be initialized by a constant expression}}
// expected-note@+1 {{bit_cast involving type 'bool __attribute__((ext_vector_type(17)))' (vector of 17 'bool' values) is not allowed in a constant expression; element size 1 * element count 17 is not a multiple of the byte size 8}}
constexpr bool17 bad_int_to_bool17 = __builtin_bit_cast(bool17, 0x0001CAFEU);
#endif
}

namespace test_complex {
  constexpr _Complex unsigned test_int_complex = { 0x0C05FEFE, 0xCAFEBABE };
  static_assert(round_trip<_Complex unsigned>(0xCAFEBABE0C05FEFEULL), "");
  static_assert(bit_cast<unsigned long long>(test_int_complex) == (LITTLE_END
                                                                   ? 0xCAFEBABE0C05FEFE
                                                                   : 0x0C05FEFECAFEBABE), "");
  static_assert(sizeof(double) == 2 * sizeof(float));
  struct TwoFloats { float A; float B; };
  constexpr _Complex float test_float_complex = {1.0f, 2.0f};
  constexpr TwoFloats TF = __builtin_bit_cast(TwoFloats, test_float_complex);
  static_assert(TF.A == 1.0f && TF.B == 2.0f);

  constexpr double D = __builtin_bit_cast(double, test_float_complex);
  constexpr int M = __builtin_bit_cast(int, test_int_complex); // both-error {{size of '__builtin_bit_cast' source type 'const _Complex unsigned int' does not match destination type 'int' (8 vs 4 bytes)}}
}


namespace OversizedBitField {
#if defined(_WIN32)
  /// This is an error (not just a warning) on Windows and the field ends up with a size of 1 instead of 4.
#else
  typedef unsigned __INT16_TYPE__ uint16_t;
  typedef unsigned __INT32_TYPE__ uint32_t;
  struct S {
    uint16_t a : 20; // both-warning {{exceeds the width of its type}}
  };

  static_assert(sizeof(S) == 4);
  static_assert(__builtin_bit_cast(S, (uint32_t)32).a == (LITTLE_END ? 32 : 0)); // ref-error {{not an integral constant expression}} \
                                                                                 // ref-note {{constexpr bit_cast involving bit-field is not yet supported}}
#endif
}

namespace Discarded {
  enum my_byte : unsigned char {};
  struct pad {
    char a;
    int b;
  };
  constexpr int bad_my_byte = (__builtin_bit_cast(my_byte[8], pad{1, 2}), 0); // both-error {{must be initialized by a constant expression}} \
                                                                              // both-note {{indeterminate value can only initialize an object of type 'unsigned char' or 'std::byte';}}
}

typedef bool bool9 __attribute__((ext_vector_type(9)));
// both-error@+2 {{constexpr variable 'bad_bool9_to_short' must be initialized by a constant expression}}
// both-note@+1 {{bit_cast involving type 'bool __attribute__((ext_vector_type(9)))' (vector of 9 'bool' values) is not allowed in a constant expression; element size 1 * element count 9 is not a multiple of the byte size 8}}
constexpr unsigned short bad_bool9_to_short = __builtin_bit_cast(unsigned short, bool9{1,1,0,1,0,1,0,1,0});

// both-warning@+2 {{returning reference to local temporary object}}
// both-note@+1 {{temporary created here}}
constexpr const intptr_t &returns_local() { return 0L; }

// both-error@+2 {{constexpr variable 'test_nullptr_bad' must be initialized by a constant expression}}
// both-note@+1 {{read of temporary whose lifetime has ended}}
constexpr nullptr_t test_nullptr_bad = __builtin_bit_cast(nullptr_t, returns_local());


#ifdef __SIZEOF_INT128__
namespace VectorCast {
  typedef unsigned X          __attribute__ ((vector_size (64)));
  typedef unsigned __int128 Y __attribute__ ((vector_size (64)));
  constexpr int test() {
    X x = {0};
    Y y = x;

    X x2 = y;

    return 0;
  }
  static_assert(test() == 0);

  typedef int X2      __attribute__ ((vector_size (64)));
  typedef __int128 Y2 __attribute__ ((vector_size (64)));
  constexpr int test2() {
    X2 x = {0};
    Y2 y = x;

    X2 x2 = y;

    return 0;
  }
  static_assert(test2() == 0);

  struct S {
    unsigned __int128 a : 3;
  };
  constexpr S s = __builtin_bit_cast(S, (__int128)12); // ref-error {{must be initialized by a constant expression}} \
                                                       // ref-note {{constexpr bit_cast involving bit-field is not yet supported}} \
                                                       // ref-note {{declared here}}
#if LITTLE_END
  static_assert(s.a == 4); // ref-error {{not an integral constant expression}} \
                           // ref-note {{initializer of 's' is not a constant expression}}
#else
  static_assert(s.a == 0); // ref-error {{not an integral constant expression}} \
                           // ref-note {{initializer of 's' is not a constant expression}}
#endif
}
#endif
