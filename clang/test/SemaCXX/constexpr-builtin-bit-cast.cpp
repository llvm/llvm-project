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

template <class T, class V> struct is_same {
  static constexpr bool value = false;
};
template <class T> struct is_same<T, T> {
  static constexpr bool value = true;
};

static_assert(sizeof(int) == 4);
static_assert(sizeof(long long) == 8);

template <class To, class From>
constexpr To bit_cast(const From &from) {
  static_assert(sizeof(To) == sizeof(From));
  // expected-note@+9 {{cannot be represented in type 'bool'}}
#ifdef __x86_64
  // expected-note@+7 {{or 'std::byte'; '__int128' is invalid}}
#endif
#ifdef __CHAR_UNSIGNED__
  // expected-note@+4 2 {{indeterminate value can only initialize an object of type 'unsigned char', 'char', or 'std::byte'; 'signed char' is invalid}}
#else
  // expected-note@+2 2 {{indeterminate value can only initialize an object of type 'unsigned char' or 'std::byte'; 'signed char' is invalid}}
#endif
  return __builtin_bit_cast(To, from);
}

template <class Intermediate, class Init>
constexpr bool round_trip(const Init &init) {
  return bit_cast<Init>(bit_cast<Intermediate>(init)) == init;
}

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

  static_assert(round_trip<unsigned long long>(splice));
  static_assert(round_trip<long long>(splice));

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
  static_assert(round_trip<tuple4>(b));
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

  constexpr pad pir{4, 4};
  // expected-error@+2 {{constexpr variable 'piw' must be initialized by a constant expression}}
  // expected-note@+1 {{in call to 'bit_cast(pir)'}}
  constexpr int piw = bit_cast<no_pad>(pir).x;

  // expected-error@+2 {{constexpr variable 'bad' must be initialized by a constant expression}}
  // expected-note@+1 {{in call to 'bit_cast(pir)'}}
  constexpr no_pad bad = bit_cast<no_pad>(pir);

  constexpr pad fine = bit_cast<pad>(no_pad{1, 2, 3, 4, 5});
  static_assert(fine.x == 1 && fine.y == 5);
}

void no_bitfields() {
  // FIXME!
  struct S {
    unsigned char x : 8;
  };

  struct G {
    unsigned char x : 8;
  };

  constexpr S s{0};
  // expected-error@+2 {{constexpr variable 'g' must be initialized by a constant expression}}
  // expected-note@+1 {{constexpr bit_cast involving bit-field is not yet supported}}
  constexpr G g = __builtin_bit_cast(G, s);
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

  static_assert(round_trip<G>(s));
  static_assert(round_trip<S>(g));
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

namespace std {
enum byte : unsigned char {};
}

enum my_byte : unsigned char {};

struct pad {
  char a;
  int b;
};

constexpr int ok_byte = (__builtin_bit_cast(std::byte[8], pad{1, 2}), 0);
constexpr int ok_uchar = (__builtin_bit_cast(unsigned char[8], pad{1, 2}), 0);

#ifdef __CHAR_UNSIGNED__
// expected-note@+5 {{indeterminate value can only initialize an object of type 'unsigned char', 'char', or 'std::byte'; 'my_byte' is invalid}}}}
#else
// expected-note@+3 {{indeterminate value can only initialize an object of type 'unsigned char' or 'std::byte'; 'my_byte' is invalid}}
#endif
// expected-error@+1 {{constexpr variable 'bad_my_byte' must be initialized by a constant expression}}
constexpr int bad_my_byte = (__builtin_bit_cast(my_byte[8], pad{1, 2}), 0);
#ifndef __CHAR_UNSIGNED__
// expected-error@+3 {{constexpr variable 'bad_char' must be initialized by a constant expression}}
// expected-note@+2 {{indeterminate value can only initialize an object of type 'unsigned char' or 'std::byte'; 'char' is invalid}}
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

constexpr bool test_bad_bool = bit_cast<bool>('A'); // expected-error {{must be initialized by a constant expression}} expected-note{{in call}}

static_assert(round_trip<signed char>(true), "");
static_assert(round_trip<unsigned char>(false), "");
static_assert(round_trip<bool>(false), "");

static_assert(round_trip<bool>((char)0), "");
static_assert(round_trip<bool>((char)1), "");
}

namespace test_long_double {
#ifdef __x86_64
constexpr __int128_t test_cast_to_int128 = bit_cast<__int128_t>((long double)0); // expected-error{{must be initialized by a constant expression}} expected-note{{in call}}

constexpr long double ld = 3.1425926539;

struct bytes {
  unsigned char d[16];
};

static_assert(round_trip<bytes>(ld), "");

static_assert(round_trip<long double>(10.0L));

constexpr bool f(bool read_uninit) {
  bytes b = bit_cast<bytes>(ld);
  unsigned char ld_bytes[10] = {
    0x0,  0x48, 0x9f, 0x49, 0xf0,
    0x3c, 0x20, 0xc9, 0x0,  0x40,
  };

  for (int i = 0; i != 10; ++i)
    if (ld_bytes[i] != b.d[i])
      return false;

  if (read_uninit && b.d[10]) // expected-note{{read of uninitialized object is not allowed in a constant expression}}
    return false;

  return true;
}

static_assert(f(/*read_uninit=*/false), "");
static_assert(f(/*read_uninit=*/true), ""); // expected-error{{static assertion expression is not an integral constant expression}} expected-note{{in call to 'f(true)'}}

constexpr bytes ld539 = {
  0x0, 0x0,  0x0,  0x0,
  0x0, 0x0,  0xc0, 0x86,
  0x8, 0x40, 0x0,  0x0,
  0x0, 0x0,  0x0,  0x0,
};

constexpr long double fivehundredandthirtynine = 539.0;

static_assert(bit_cast<long double>(ld539) == fivehundredandthirtynine, "");

#else
static_assert(round_trip<__int128_t>(34.0L));
#endif
}
