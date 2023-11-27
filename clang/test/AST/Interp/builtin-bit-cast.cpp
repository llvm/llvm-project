// RUN: %clang_cc1 -verify -std=c++2a -fsyntax-only -fexperimental-new-constant-interpreter %s
// RUN: %clang_cc1 -verify=ref -std=c++2a -fsyntax-only %s
// RUN: %clang_cc1 -verify -std=c++2a -fsyntax-only -triple aarch64_be-linux-gnu -fexperimental-new-constant-interpreter %s
// RUN: %clang_cc1 -verify=ref -std=c++2a -fsyntax-only -triple aarch64_be-linux-gnu %s
// RUN: %clang_cc1 -verify -std=c++2a -fsyntax-only -fexperimental-new-constant-interpreter -triple powerpc64le-unknown-unknown -mabi=ieeelongdouble %s
// RUN: %clang_cc1 -verify=ref -std=c++2a -fsyntax-only -triple powerpc64le-unknown-unknown -mabi=ieeelongdouble %s
// RUN: %clang_cc1 -verify -std=c++2a -fsyntax-only -fexperimental-new-constant-interpreter -triple powerpc64-unknown-unknown -mabi=ieeelongdouble %s
// RUN: %clang_cc1 -verify=ref -std=c++2a -fsyntax-only -triple powerpc64-unknown-unknown -mabi=ieeelongdouble %s

/// FIXME: This is a version of
///   clang/test/SemaCXX/constexpr-builtin-bit-cast.cpp with the currently
///   supported subset of operations. They should *all* be supported though.


#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
#  define LITTLE_END 1
#elif __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#  define LITTLE_END 0
#else
#  error "huh?"
#endif

typedef decltype(nullptr) nullptr_t;



static_assert(sizeof(int) == 4);
static_assert(sizeof(long long) == 8);

template <class To, class From>
constexpr To bit_cast(const From &from) {
  static_assert(sizeof(To) == sizeof(From));
  return __builtin_bit_cast(To, from); // ref-note 2{{indeterminate value can only initialize}} \
                                       // expected-note 2{{indeterminate value can only initialize}} \
                                       // ref-note {{subexpression not valid}}
}


/// Current interpreter does not support this.
/// https://github.com/llvm/llvm-project/issues/63686
constexpr int FromString = bit_cast<int>("abc"); // ref-error {{must be initialized by a constant expression}} \
                                                 // ref-note {{in call to}} \
                                                 // ref-note {{declared here}}
#if LITTLE_END
static_assert(FromString == 6513249); // ref-error {{is not an integral constant expression}} \
                                      // ref-note {{initializer of 'FromString' is not a constant expression}}
#else
static_assert(FromString == 1633837824); // ref-error {{is not an integral constant expression}} \
                                         // ref-note {{initializer of 'FromString' is not a constant expression}}
#endif


struct S {
  int i, j, k;
};
constexpr S func() {
  constexpr int array[] = { 12, 42, 128 };
  return __builtin_bit_cast(S, array);
}
constexpr S s = func();
static_assert(s.i == 12, "");
static_assert(s.j == 42, "");
static_assert(s.k == 128, "");

template <class Intermediate, class Init>
constexpr bool round_trip(const Init &init) {
  return bit_cast<Init>(bit_cast<Intermediate>(init)) == init;
}

/// We can ignore it.
constexpr int foo() {
  (void)__builtin_bit_cast(int, 3);
  return 1;
}
static_assert(foo() == 1, "");


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

namespace Ints {
  static_assert(round_trip<unsigned>((int)-1));
  static_assert(round_trip<unsigned>((int)0x12345678));
  static_assert(round_trip<unsigned>((int)0x87654321));
  static_assert(round_trip<unsigned>((int)0x0C05FEFE));
  static_assert(round_trip<float>((int)0x0C05FEFE));
}

namespace FloatToDouble {
  constexpr float F1[] = {1.0f, 2.0f};
  constexpr double D1 = __builtin_bit_cast(double, F1);
  static_assert(D1 > 0);
}

namespace Arrays {
  constexpr unsigned char input[] = {0xCA, 0xFE, 0xBA, 0xBE};
  constexpr unsigned expected = LITTLE_END ? 0xBEBAFECA : 0xCAFEBABE;
  static_assert(bit_cast<unsigned>(input) == expected);

  constexpr short S[] = {10, 20};
  constexpr int I = __builtin_bit_cast(int, S);
  static_assert(I == (LITTLE_END ? 1310730 : 655380));
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

static_assert(bit_cast<unsigned long long>(splice) == (LITTLE_END
                                                           ? 0xCAFEBABE0C05FEFE
                                                           : 0x0C05FEFECAFEBABE));

constexpr int_splicer IS = bit_cast<int_splicer>(0xCAFEBABE0C05FEFE);
static_assert(bit_cast<int_splicer>(0xCAFEBABE0C05FEFE).x == (LITTLE_END
                                                                  ? 0x0C05FEFE
                                                                  : 0xCAFEBABE));

static_assert(round_trip<unsigned long long>(splice));
static_assert(round_trip<long long>(splice));

struct base2 {
};

struct base3 {
  unsigned z;
  constexpr base3() : z(3) {}
};

struct bases : int_splicer, base2, base3 {
  unsigned doublez;
  constexpr bases() : doublez(4) {}
};

struct tuple4 {
  unsigned x, y, z, doublez;

  constexpr bool operator==(tuple4 const &other) const {
    return x == other.x && y == other.y &&
           z == other.z && doublez == other.doublez;
  }
};
constexpr bases b;// = {{1, 2}, {}, {3}, 4};
constexpr tuple4 t4 = bit_cast<tuple4>(b);

// Regardless of endianness, this should hold:
static_assert(t4.x == 1);
static_assert(t4.y == 2);
static_assert(t4.z == 3);
static_assert(t4.doublez == 4);
static_assert(t4 == tuple4{1, 2, 3, 4});
static_assert(round_trip<tuple4>(b));

namespace WithBases {
  struct Base {
    char A[3] = {1,2,3};
  };

  struct A : Base {
    char B = 12;
  };

  constexpr A a;
  constexpr unsigned I = __builtin_bit_cast(unsigned, a);
  static_assert(I == (LITTLE_END ? 201523713 : 16909068));
};



void test_array() {
  constexpr unsigned char input[] = {0xCA, 0xFE, 0xBA, 0xBE};
  constexpr unsigned expected = LITTLE_END ? 0xBEBAFECA : 0xCAFEBABE;
  static_assert(bit_cast<unsigned>(input) == expected);
}


namespace test_array_fill {
  constexpr unsigned char a[4] = {1, 2};
  constexpr unsigned int i = bit_cast<unsigned int>(a);
  static_assert(i == (LITTLE_END ? 0x00000201 : 0x01020000));
}

namespace Another {
  constexpr char C[] = {1,2,3,4};
  struct F{ short a; short b; };
  constexpr F f = __builtin_bit_cast(F, C);

#if LITTLE_END
  static_assert(f.a == 513);
  static_assert(f.b == 1027);
#else
  static_assert(f.a == 258);
  static_assert(f.b == 772);
#endif
}


namespace array_members {
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

namespace CompositeArrays {
  struct F {
    int a;
    int b;
  };

  static_assert(sizeof(long) == 2 * sizeof(int));

  constexpr F ff[] = {{1,2}};
  constexpr long L = __builtin_bit_cast(long, ff);

#if LITTLE_END
  static_assert(L == 8589934593);
#else
  static_assert(L == 4294967298);
#endif
}



#ifdef __CHAR_UNSIGNED__
// ref-note@+5 {{indeterminate value can only initialize an object of type 'unsigned char', 'char', or 'std::byte'; 'unsigned long' is invalid}}
#else
// ref-note@+3 {{indeterminate value can only initialize an object of type 'unsigned char' or 'std::byte'; 'unsigned long' is invalid}}
#endif
// ref-error@+1 {{constexpr variable 'test_from_nullptr' must be initialized by a constant expression}}
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

namespace IndeterminateToPrimitive {
  struct S {
    bool a;
    // One byte of padding.
    short b;
  };
  constexpr S s{true, 12};

  static_assert(sizeof(S) == sizeof(int), "");
  constexpr int A = __builtin_bit_cast(int, s); // ref-error {{must be initialized by a constant expression}} \
                                                // ref-note {{indeterminate value}} \
                                                // expected-error {{must be initialized by a constant expression}} \
                                                // expected-note {{indeterminate value}}
}

namespace test_partially_initialized {
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
  // ref-error@-1 {{constexpr variable 'piw' must be initialized by a constant expression}}
  // ref-note@-2 {{in call to}}

  // expected-error@+2 {{constexpr variable 'bad' must be initialized by a constant expression}}
  // expected-note@+1 {{in call to 'bit_cast(pir)'}}
  constexpr no_pad bad = bit_cast<no_pad>(pir);
  // ref-error@-1 {{constexpr variable 'bad' must be initialized by a constant expression}}
  // ref-note@-2 {{in call to}}

  constexpr pad fine = bit_cast<pad>(no_pad{1, 2, 3, 4, 5});
  static_assert(fine.x == 1 && fine.y == 5);
}

namespace bad_types {
  union X {
    int x;
  };

  struct G {
    int g;
  };
  // ref-error@+2 {{constexpr variable 'g' must be initialized by a constant expression}}
  // ref-note@+1 {{bit_cast from a union type is not allowed in a constant expression}}
  constexpr G g = __builtin_bit_cast(G, X{0});
  // expected-error@-1 {{constexpr variable 'g' must be initialized by a constant expression}}
  // expected-note@-2 {{bit_cast from a union type is not allowed in a constant expression}}

  // ref-error@+2 {{constexpr variable 'x' must be initialized by a constant expression}}
  // ref-note@+1 {{bit_cast to a union type is not allowed in a constant expression}}
  constexpr X x = __builtin_bit_cast(X, G{0});
  // expected-error@-1 {{constexpr variable 'x' must be initialized by a constant expression}}
  // expected-note@-2 {{bit_cast to a union type is not allowed in a constant expression}}

  struct has_pointer {
    int *ptr; // ref-note 2{{invalid type 'int *' is a member of}} \
              // expected-note 2{{invalid type 'int *' is a member of}}
  };

  // ref-error@+2 {{constexpr variable 'ptr' must be initialized by a constant expression}}
  // ref-note@+1 {{bit_cast from a pointer type is not allowed in a constant expression}}
  constexpr unsigned long ptr = __builtin_bit_cast(unsigned long, has_pointer{0});
  // expected-error@-1 {{constexpr variable 'ptr' must be initialized by a constant expression}}
  // expected-note@-2 {{bit_cast from a pointer type is not allowed in a constant expression}}


  // ref-error@+2 {{constexpr variable 'hptr' must be initialized by a constant expression}}
  // ref-note@+1 {{bit_cast to a pointer type is not allowed in a constant expression}}
  constexpr has_pointer hptr =  __builtin_bit_cast(has_pointer, 0ul);
  // expected-error@-1 {{constexpr variable 'hptr' must be initialized by a constant expression}}
  // expected-note@-2 {{bit_cast to a pointer type is not allowed in a constant expression}}
}

namespace backtrace {
  struct A {
    int *ptr; // expected-note {{invalid type 'int *' is a member of 'backtrace::A'}} \
              // ref-note {{invalid type 'int *' is a member of 'backtrace::A'}}
  };

  struct B {
    A as[10]; // expected-note {{invalid type 'A[10]' is a member of 'backtrace::B'}} \
              // ref-note {{invalid type 'A[10]' is a member of 'backtrace::B'}}
  };

    struct C : B { // expected-note {{invalid type 'B' is a base of 'backtrace::C'}} \
                   // ref-note {{invalid type 'B' is a base of 'backtrace::C'}}
  };

  struct E {
    unsigned long ar[10];
  };

  constexpr E e = __builtin_bit_cast(E, C{}); // expected-error {{must be initialized by a constant expression}} \
                                              // expected-note {{bit_cast from a pointer type is not allowed}} \
                                              // ref-error {{must be initialized by a constant expression}} \
                                              // ref-note {{bit_cast from a pointer type is not allowed}}
}

namespace ReferenceMember {
  struct ref_mem {
    const int &rm;
  };

  typedef unsigned long ulong;
  constexpr int global_int = 0;

   constexpr ulong run_ref_mem = __builtin_bit_cast(ulong, ref_mem{global_int}); // ref-error {{must be initialized by a constant expression}} \
                                                                                 // ref-note {{bit_cast from a type with a reference member}} \
                                                                                 // expected-error {{must be initialized by a constant expression}} \
                                                                                 // expected-note {{bit_cast from a type with a reference member}}
}

namespace FromUnion {
  union u {
    int im;
  };

  constexpr int run_u = __builtin_bit_cast(int, u{32}); // ref-error {{must be initialized by a constant expression}} \
                                                        // ref-note {{bit_cast from a union type}} \
                                                        // expected-error {{must be initialized by a constant expression}} \
                                                        // expected-note {{bit_cast from a union type}}
}


struct vol_mem {
   volatile int x; // expected-note {{invalid type 'volatile int' is a member of 'vol_mem'}}
};

namespace VolatileMember {
  constexpr int run_vol_mem = __builtin_bit_cast(int, vol_mem{43}); // ref-error {{must be initialized by a constant expression}} \
                                                                    // ref-note {{non-literal type 'vol_mem'}} \
                                                                    // expected-error {{must be initialized by a constant expression}} \
                                                                    // expected-note {{bit_cast from a volatile type}}
}

namespace MemberPointer {
  /// FIXME: The diagnostic for bitcasts is properly implemented, but we lack support for member pointers.
#if 0
  struct mem_ptr {
    int vol_mem::*x; // expected-note{{invalid type 'int vol_mem::*' is a member of 'mem_ptr'}}
  };
  constexpr int run_mem_ptr = __builtin_bit_cast(unsigned long, mem_ptr{nullptr}); // ref-error {{must be initialized by a constant expression}} \
                                                                                   // ref-note {{bit_cast from a member pointer type}} \
                                                                                   // expected-error {{must be initialized by a constant expression}} \
                                                                                   // expected-note {{bit_cast from a member pointer type}}
#endif
}

  struct A { char c; /* char padding : 8; */ short s; };
  struct B { unsigned char x[4]; };

  constexpr B one() {
    A a = {1, 2};
    return bit_cast<B>(a);
  }
  /// FIXME: The following tests need the InitMap changes.
#if 0
  constexpr char good_one = one().x[0] + one().x[2] + one().x[3];
  // ref-error@+2 {{constexpr variable 'bad_one' must be initialized by a constant expression}}
  // ref-note@+1 {{read of uninitialized object is not allowed in a constant expression}}
  constexpr char bad_one = one().x[1];
  // expected-error@-1 {{constexpr variable 'bad_one' must be initialized by a constant expression}}
  // expected-note@-2 {{read of uninitialized object is not allowed in a constant expression}}
#endif

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
  // ref-note@+5 {{indeterminate value can only initialize an object of type 'unsigned char', 'char', or 'std::byte'; 'my_byte' is invalid}}}}
#else
  // ref-note@+3 {{indeterminate value can only initialize an object of type 'unsigned char' or 'std::byte'; 'my_byte' is invalid}}
#endif
  // ref-error@+1 {{must be initialized by a constant expression}}
  constexpr int bad_my_byte = (__builtin_bit_cast(my_byte[8], pad{1, 2}), 0); // {{indeterminate value can only initialize an object of type 'unsigned char' or 'std::byte'; 'my_byte' is invalid}}

#ifndef __CHAR_UNSIGNED__
  // ref-error@+3 {{must be initialized by a constant expression}}
  // ref-note@+2 {{indeterminate value can only initialize an object of type 'unsigned char' or 'std::byte'; 'char' is invalid}}
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

namespace ValueRepr {
  /// FIXME: This is broken.
  constexpr bool b = __builtin_bit_cast(bool, (char)123);  // ref-error {{must be initialized by a constant expression}} \
                                                           // ref-note {{value 123 cannot be represented in type 'bool'}}
}

namespace FloatMember {
  struct A {
    float a;
  };
  struct B {
    unsigned char a[4];
  };

  constexpr B b = __builtin_bit_cast(B, A{1.0});

#if LITTLE_END
  static_assert(b.a[0] == 0, "");
  static_assert(b.a[1] == 0, "");
  static_assert(b.a[2] == 128, "");
  static_assert(b.a[3] == 63, "");
#else
  static_assert(b.a[0] == 63, "");
  static_assert(b.a[1] == 128, "");
  static_assert(b.a[2] == 0, "");
  static_assert(b.a[3] == 0, "");
#endif

  constexpr A a = __builtin_bit_cast(A, B{{1 << 6, 0, 0, 1 << 6}});
  static_assert(static_cast<int>(a.a) == 2, "");
}

namespace Misc {
  struct A {
    decltype(nullptr) a;
  };
  /// Not sure why this doesn't work in the current interpreter; GCC accepts it.
  constexpr unsigned long long L = __builtin_bit_cast(unsigned long long, A{nullptr}); // ref-error {{must be initialized by a constant expression}} \
                                                                                       // ref-note {{indeterminate value}}

  /// Bitcast into a nullptr_t field.
  struct B {
    unsigned char a[8];
  };

  constexpr A a = __builtin_bit_cast(A, B{{0, 0, 0, 0, 0, 0, 0, 0}});
  static_assert(a.a == nullptr, "");

  /// Uninitialized local variable bitcast'ed to a uchar.
  constexpr int primUChar() {
    signed char A;
    unsigned char B = __builtin_bit_cast(unsigned char, A);
    /// FIXME: The new interpreter doesn't print the proper diagnostic here; The read from B
    ///   should be uninitialized, since the bitcast returns indeterminate bits. However,
    ///   the code uses primitive values and those are always initialized.
    return B; // ref-note {{read of uninitialized object}}
  }
  static_assert(primUChar() == 0, ""); // ref-error {{not an integral constant expression}} \
                                       // ref-note {{in call to}} \
                                       // expected-error {{not an integral constant expression}}

  constexpr int primUChar2() {
    signed char A;
    unsigned char B = __builtin_bit_cast(unsigned char, A) + 1;
    /// Same problem as above, but the diagnostic of the current interpreter is just as bad as ours.
    return B;
  }
  static_assert(primUChar2() == 0, ""); // ref-error {{not an integral constant expression}} \
                                        // expected-error {{not an integral constant expression}}

  /// This time, the uchar is in a struct.
  constexpr int primUChar3() {
    struct B {
      unsigned char b;
    };
    signed char A;
    B b = __builtin_bit_cast(B, A);
    return b.b; // ref-note {{read of uninitialized object}} \
                // expected-note {{read of uninitialized object}}
  }
  static_assert(primUChar3() == 0, ""); // ref-error {{not an integral constant expression}} \
                                        // ref-note {{in call to}} \
                                        // expected-error {{not an integral constant expression}} \
                                        // expected-note {{in call to}}

  /// Fully initialized local variable that should end up being un-initalized again because the
  /// bit cast returns bits of indeterminate value.
  constexpr int primUChar4() {
    unsigned char c = 'a';
    signed char cu;

    c = __builtin_bit_cast(unsigned char, cu);

    return c; // ref-note {{read of uninitialized object}}
  }
  static_assert(primUChar4() == 0, ""); // ref-error {{not an integral constant expression}} \
                                        // ref-note {{in call to}} \
                                        // expected-error {{not an integral constant expression}}


  /// Casting an uninitialized struct.
  constexpr int primUChar5(bool DoBC) {
    struct A {
      unsigned char a; // ref-note {{subobject declared here}}
    };
    struct B {
      signed char b;
    };

    A a = {12};

    if (DoBC) {
      B b;
      a = __builtin_bit_cast(A, b); // expected-note {{in call to}} \
                                    // expected-note {{read of uninitialized object}} \
                                    // ref-note {{subobject 'a' is not initialized}} \
                                    // ref-note {{in call to}}
    }

    return a.a;
  }
  static_assert(primUChar5(false) == 12, "");
  static_assert(primUChar5(true) == 12, ""); // expected-error {{not an integral constant expression}} \
                                             // expected-note {{in call to 'primUChar5(true)'}} \
                                             // ref-error {{not an integral constant expression}} \
                                             // ref-note {{in call to 'primUChar5(true)'}}
}


constexpr unsigned char identity1a = 42;
constexpr unsigned char identity1b = __builtin_bit_cast(unsigned char, identity1a);
static_assert(identity1b == 42);

#ifdef __PPC64__
namespace LongDouble {
  struct bytes {
    unsigned char d[sizeof(long double)];
  };

  constexpr long double ld = 3.1425926539;
  constexpr long double ldmax = __LDBL_MAX__;
  static_assert(round_trip<bytes>(ld), "");
  static_assert(round_trip<bytes>(ldmax), "");

  constexpr bytes b = __builtin_bit_cast(bytes, ld);

#if LITTLE_END
static_assert(b.d[0] == 0);
static_assert(b.d[1] == 0);
static_assert(b.d[2] == 0);
static_assert(b.d[3] == 0);
static_assert(b.d[4] == 0);
static_assert(b.d[5] == 0);

static_assert(b.d[6] == 0);
static_assert(b.d[7] == 144);
static_assert(b.d[8] == 62);
static_assert(b.d[9] == 147);
static_assert(b.d[10] == 224);
static_assert(b.d[11] == 121);
static_assert(b.d[12] == 64);
static_assert(b.d[13] == 146);
static_assert(b.d[14] == 0);
static_assert(b.d[15] == 64);
#else
static_assert(b.d[0] == 64);
static_assert(b.d[1] == 0);
static_assert(b.d[2] == 146);
static_assert(b.d[3] == 64);
static_assert(b.d[4] == 121);
static_assert(b.d[5] == 224);
static_assert(b.d[6] == 147);
static_assert(b.d[7] == 62);
static_assert(b.d[8] == 144);
static_assert(b.d[9] == 0);

static_assert(b.d[10] == 0);
static_assert(b.d[11] == 0);
static_assert(b.d[12] == 0);
static_assert(b.d[13] == 0);
static_assert(b.d[14] == 0);
static_assert(b.d[15] == 0);
#endif
}
#endif // __PPC64__

#ifdef __x86_64
namespace LongDoubleX86 {
  struct bytes {
    unsigned char d[16]; // ref-note {{declared here}}
  };

  constexpr long double ld = 3.1425926539;
  static_assert(round_trip<bytes>(ld), "");

  /// The current interpreter rejects this (probably because the APFloat only
  /// uses 10 bytes to represent the value instead of the full 16 that the long double
  /// takes up). MSVC and GCC accept it though.
  constexpr bytes b = __builtin_bit_cast(bytes, ld); // ref-error {{must be initialized by a constant expression}} \
                                                     // ref-note {{subobject 'd' is not initialized}}
}
#endif

namespace StringLiterals {
  template<int n>
  struct StrBuff {
    char data[n];
  };

  constexpr StrBuff<4> Foo = __builtin_bit_cast(StrBuff<4>, "foo"); // ref-error {{must be initialized by a constant expression}} \
                                                                    // ref-note 4{{declared here}}
  static_assert(Foo.data[0] == 'f', "");  // ref-error {{not an integral constant expression}} \
                                          // ref-note {{initializer of 'Foo' is not a constant expression}}
  static_assert(Foo.data[1] == 'o', "");  // ref-error {{not an integral constant expression}} \
                                          // ref-note {{initializer of 'Foo' is not a constant expression}}
  static_assert(Foo.data[2] == 'o', "");  // ref-error {{not an integral constant expression}} \
                                          // ref-note {{initializer of 'Foo' is not a constant expression}}
  static_assert(Foo.data[3] == '\0', ""); // ref-error {{not an integral constant expression}} \
                                          // ref-note {{initializer of 'Foo' is not a constant expression}}
};

/// The current interpreter does not support bitcasts involving bitfields at all,
/// so the following is mainly from comparing diagnostic output with GCC.
namespace Bitfields {
  struct S {
    char a : 8;
  };

  constexpr S s{4};
  constexpr char c = __builtin_bit_cast(char, s); // ref-error {{must be initialized by a constant expression}} \
                                                  // ref-note {{bit_cast involving bit-field is not yet supported}} \
                                                  // ref-note{{declared here}}
  static_assert(c == 4, ""); // ref-error {{not an integral constant expression}} \
                             // ref-note {{initializer of 'c' is not a constant expression}}


  struct S2 {
    char a : 4;
  };
  constexpr S2 s2{4};
  constexpr char c2 = __builtin_bit_cast(char, s2); // expected-error {{must be initialized by a constant expression}} \
                                                    // expected-note {{indeterminate value can only initialize an object of type 'unsigned char' or 'std::byte'; 'char' is invalid}} \
                                                    // ref-error {{must be initialized by a constant expression}} \
                                                    // ref-note {{bit_cast involving bit-field is not yet supported}}

  struct A {
    unsigned char a : 4;
    unsigned char b : 4;
  };

  constexpr A b{12, 3};
  static_assert(b.a == 12, "");
  static_assert(b.b == 3, "");
  constexpr unsigned char a = __builtin_bit_cast(unsigned char, b); // ref-error {{must be initialized by a constant expression}} \
                                                                    // ref-note {{bit_cast involving bit-field is not yet supported}} \
                                                                    // ref-note {{declared here}}
  static_assert(a == 60, ""); // ref-error {{not an integral constant expression}} \
                              // ref-note {{initializer of 'a' is not a constant expression}}

  struct Byte {
    unsigned char a : 1;
    unsigned char b : 1;
    unsigned char c : 1;
    unsigned char d : 1;
    unsigned char e : 1;
    unsigned char f : 1;
    unsigned char g : 1;
    unsigned char h : 1;
  };

  constexpr Byte B = {1, 1, 0, 1, 1, 0, 0, 1};
  constexpr char C = __builtin_bit_cast(char, B); // ref-error {{must be initialized by a constant expression}} \
                                                  // ref-note {{bit_cast involving bit-field is not yet supported}} \
                                                  // ref-note {{declared here}}

  static_assert(C == -101); // ref-error {{not an integral constant expression}} \
                            // ref-note {{initializer of 'C' is not a constant expression}}

  struct P {
    unsigned short s1 : 5;
    short s2;
  };

  constexpr P p = {24, -10};
  constexpr int I = __builtin_bit_cast(int, p); // ref-error {{must be initialized by a constant expression}} \
                                                // ref-note {{bit_cast involving bit-field is not yet supported}} \
                                                // expected-error {{must be initialized by a constant expression}} \
                                                // expected-note {{indeterminate value}}


  struct CharStruct {
    unsigned char v;
  };
  constexpr CharStruct CS = __builtin_bit_cast(CharStruct, B);  // ref-error {{must be initialized by a constant expression}} \
                                                                // ref-note {{bit_cast involving bit-field is not yet supported}} \
                                                                // ref-note {{declared here}}
  static_assert(CS.v == 155); // ref-error {{not an integral constant expression}} \
                              // ref-note {{initializer of 'CS' is not a constant expression}}


  struct I3 {
    int a;
    int b : 10;
    int c;
  };

  struct I32 {
    int a;
    int b;
    int c;
  };

  constexpr I3 i3 {5, 10, 15};
  constexpr I32 i32 = __builtin_bit_cast(I32, i3); // ref-error {{must be initialized by a constant expression}} \
                                                   // ref-note {{bit_cast involving bit-field is not yet supported}} \
                                                   // expected-error {{must be initialized by a constant expression}} \
                                                   // expected-note {{indeterminate value can only initialize an object of type 'unsigned char'}}

  struct I33 {
    int a;
    unsigned char b;
    int c;
  };

  constexpr I33 i33 = __builtin_bit_cast(I33, i3); // ref-error {{must be initialized by a constant expression}} \
                                                   // ref-note {{bit_cast involving bit-field is not yet supported}} \
                                                   // ref-note 3{{declared here}}
  static_assert(i33.a == 5, ""); // ref-error {{not an integral constant expression}} \
                                 // ref-note {{initializer of 'i33' is not a constant expression}}
  static_assert(i33.b == 10, ""); // ref-error {{not an integral constant expression}} \
                                  // ref-note {{initializer of 'i33' is not a constant expression}}
  static_assert(i33.c == 15, ""); // ref-error {{not an integral constant expression}} \
                                  // ref-note {{initializer of 'i33' is not a constant expression}}
}
