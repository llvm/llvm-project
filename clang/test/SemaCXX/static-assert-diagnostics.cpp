// RUN: %clang_cc1 -std=c++2a -verify -fsyntax-only -triple wasm32 %s
// RUN: %clang_cc1 -std=c++2a -verify -fsyntax-only -triple aarch64_be %s
// RUN: %clang_cc1 -std=c++2a -verify -fsyntax-only -triple x86_64 -DTEST_CLIP %s
// RUN: %clang_cc1 -std=c++2a -verify -fsyntax-only -triple x86_64 -DTEST_CLIP=SMALL -fconstexpr-print-value-size-limit=60 %s
// RUN: %clang_cc1 -std=c++2a -verify -fsyntax-only -triple x86_64 -DTEST_CLIP=NO_LIMIT -fconstexpr-print-value-size-limit=0 %s

struct A {
  int a, b[3], c;
  bool operator==(const A&) const = default;
};

constexpr auto a0 = A{0, 0, 3, 4, 5};

// expected-note@+1 {{evaluates to '(const A){0, {0, 3, 4}, 5} == A{1, {2, 3, 4}, 5}'}}
static_assert(a0 == A{1, {2, 3, 4}, 5}); // expected-error {{failed}}

// `operator==` wrapper type
struct _arr {
  const int b[3];
  constexpr bool operator==(const int rhs[3]) const {
    for (unsigned i = 0; i < sizeof(b) / sizeof(int); i++)
      if (b[i] != rhs[i])
        return false;
    return true;
  }
};

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc99-extensions"
static_assert(_arr{{2, 3, 4}} == (const int[3]){2, 3, 4});
#pragma clang diagnostic pop

// expected-note@+1 {{{evaluates to '_arr{{2, 3, 4}} == (const int[3]){0, 3, 4}'}}}
static_assert(_arr{2, 3, 4} == a0.b); // expected-error {{failed}}

struct B {
  int a, c; // named the same just to keep things fresh
  bool operator==(const B&) const = default;
};

// expected-note@+1 {{evaluates to 'B{7, 6} == B{8, 6}'}}
static_assert(B{7, 6} == [] { return B{8, 6}; }()); // expected-error {{failed}}

typedef int v4si __attribute__((__vector_size__(16)));

struct C: A, B {
  enum { E1, E2 } e;
  bool operator==(const C&) const = default;
};

constexpr auto cc = C{A{1, {2, 3, 4}, 5}, B{7, 6}, C::E1};

// expected-note@+1 {{{evaluates to '(const C){{1, {2, 3, 4}, 5}, {7, 6}, C::E1} == C{{0, {0, 3, 4}, 5}, {5, 0}, C::E2}'}}}
static_assert(cc == C{a0, {5}, C::E2}); // expected-error {{failed}}

enum E { numerator };
constexpr E e = E::numerator;
static_assert(numerator == ((E)0));
static_assert(((E)0) == ((E)7)); // expected-error {{failed}}
// expected-note@-1 {{{evaluates to 'numerator == (E)7'}}}

typedef enum { something } MyEnum;
static_assert(MyEnum::something == ((MyEnum)7)); // expected-error {{failed}}
// expected-note@-1 {{{evaluates to 'something == (MyEnum)7'}}}

// unnamed enums
static_assert(C::E1 == (decltype(C::e))0);
// expected-note@+1 {{{evaluates to 'C::E1 == C::E2'}}}
static_assert(C::E1 == (decltype(C::e))1); // expected-error {{failed}}
static_assert(C::E1 == (decltype(C::e))7); // expected-error {{failed}}
// expected-note@-1 {{{evaluates to 'C::E1 == (decltype(C::e))7'}}}

constexpr enum { declLocal } ee = declLocal;
static_assert(((decltype(ee))0) == ee);
static_assert(((decltype(ee))0) == ((decltype(ee))7)); // expected-error {{failed}}
// expected-note@-1 {{{evaluates to 'declLocal == (decltype(ee))7'}}}

struct TU {
  enum { S, U } Tag;
  union {
    signed int s;
    unsigned int u;
  };
  constexpr bool operator==(const TU& rhs) const {
    if (Tag != rhs.Tag) return false;
    switch (Tag) {
      case S:
        return s == rhs.s;
      case U:
        return u == rhs.u;
    }
  };
};
static_assert(TU{TU::S, {7}} == TU{TU::S, {.s=7}});
static_assert(TU{TU::U, {.u=9}} == TU{TU::U, {.u=9}});

// expected-note@+1 {{{evaluates to 'TU{TU::S, {.s = 7}} == TU{TU::S, {.s = 6}}'}}}
static_assert(TU{TU::S, {.s=7}} == TU{TU::S, {.s=6}}); // expected-error {{failed}}
static_assert(TU{TU::U, {.u=7}} == TU{TU::U, {.u=9}}); // expected-error {{failed}}
// expected-note@-1 {{{evaluates to 'TU{TU::U, {.u = 7}} == TU{TU::U, {.u = 9}}'}}}

struct EnumArray {
  const E nums[3];
  constexpr bool operator==(const E rhs[3]) const {
    for (unsigned i = 0; i < sizeof(nums) / sizeof(E); i++)
      if (nums[i] != rhs[i])
        return false;
    return true;

  };
};
static_assert(EnumArray{} == (const E[3]){numerator});

// expected-note@+1 {{{evaluates to 'EnumArray{{}} == (const E[3]){numerator, (const E)1, (const E)2}'}}}
static_assert(EnumArray{} == (const E[3]){(E)0, (E)1, (E)2}); // expected-error {{failed}}

// define `std::bit_cast`
namespace std {
template <class To, class From>
constexpr To bit_cast(const From &from) {
  static_assert(sizeof(To) == sizeof(From));
  return __builtin_bit_cast(To, from);
}
} // namespace std

namespace vector {
typedef int v4si __attribute__((__vector_size__(16)));

struct V {
  v4si v;

  // doesn't work
  // vectors are not contextually convertable to `bool`, and
  // `==` on vectors produces a vector of element-wise results
  // bool operator==(const V&) const = default;

  constexpr bool operator==(const V& rhs) const {
    // doesn't work
    // __builtin_reduce_and is not valid in a constant expression
    // return __builtin_reduce_and(b == rhs.b) && __builtin_reduce_and(v == rhs.v);

    // also doesn't work
    // surprisingly, b[0] is also not valid in a constant expression (nor v[0])
    // return b[0] == rhs.b[0] && ...

    // cmp an array of bytes that does element-wise comparisons that's the same size as v
    struct cmp {
      unsigned char v [sizeof(v4si)];
      bool operator==(const cmp&) const = default;
    };
    return std::bit_cast<cmp>(v) == std::bit_cast<cmp>(rhs.v);
  };
};
constexpr bool operator==(const V& lhs, const v4si& rhs) {
  return lhs == V{rhs};
}

constexpr auto vv = V{1, 2, 3, 4};

static_assert(V{1, 2, 3, 4} == V{1, 2, 3, 4});

// expected-note@+1 {{{evaluates to 'V{{1, 2, 3, 4}} == V{{1, 2, 0, 4}}'}}}
static_assert(V{1, 2, 3, 4} == [] { return V{1, 2, 0, 4}; }()); // expected-error {{failed}}
// expected-note@+1 {{{evaluates to 'V{{1, 2, 3, 4}} == (v4si){1, 2, 0, 4}'}}}
static_assert(V{1, 2, 3, 4} == [] { return (v4si){1, 2, 0, 4}; }()); // expected-error {{failed}}

// there appears to be no constexpr-compatible way to write an == for
// two `bool4`s at this time, since std::bit_cast doesn't support it
// typedef bool bool4 __attribute__((ext_vector_type(4)));

// so we use a bool8
typedef bool bool8 __attribute__((ext_vector_type(8)));

struct BV {
  bool8 b;
  constexpr bool operator==(const BV& rhs) const {
    return std::bit_cast<unsigned char>(b) == std::bit_cast<unsigned char>(rhs.b);
  }
};
constexpr bool operator==(const BV& lhs, const bool8& rhs) {
  return lhs == BV{rhs};
}

// expected-note@+1 {{{evaluates to 'BV{{false, true, false, false, false, false, false, false}} == BV{{true, false, false, false, false, false, false, false}}'}}}
static_assert(BV{{0, 1}} == BV{{1, 0}}); // expected-error {{failed}}

// expected-note@+1 {{{evaluates to 'BV{{false, true, false, false, false, false, false, false}} == (bool8){true, false, false, false, false, false, false, false}'}}}
static_assert(BV{{0, 1}} == (bool8){true, false}); // expected-error {{failed}}
} // namespace vector

namespace {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
constexpr auto bits = 0x030201;
#elif __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
constexpr auto bits = 0x01020300;
#else
#error "don't know what to do with mixed endianness"
#endif

struct alignas(decltype(bits)) S {
unsigned char a, b, c;
};
// confusing `==` on purpose
constexpr bool operator==(const S&, const S&) { return false; }

// the note should clearly implicate the `==` implementation
// expected-error@+2 {{static assertion failed due to requirement '(anonymous namespace)::S{1, 2, 3} == std::bit_cast(bits)'}}
// expected-note@+1 {{evaluates to 'S{1, 2, 3} == S{1, 2, 3}'}}
static_assert(S{1, 2, 3} == std::bit_cast<S>(bits));

// but there should be no redundant notes
// expected-error@+1 {{static assertion failed due to requirement '(anonymous namespace)::S{1, 2, 3} == (anonymous namespace)::S{1, 2, 3}'}}
static_assert(S{1, 2, 3} == S{1, 2, 3});

// more examples of notes considered "non-redundant"
// expected-note@+1 {{evaluates to 'S{1, 2, 0} == S{1, 2, 0}'}}
static_assert(S{1, 2} == S{1, 2}); // expected-error {{failed}}
// expected-note@+1 {{evaluates to 'S{1, 2, 3} == S{1, 2, 3}'}}
static_assert(S{1, 2, 3} == S{1 + 0, 2, 3}); // expected-error {{failed}}
// expected-note@+1 {{evaluates to 'S{1, 2, 3} == S{1, 2, 3}'}}
static_assert(S{1, 2, 3} == S{1 << 0, 2, 3}); // expected-error {{failed}}
// expected-note@+1 {{evaluates to 'S{1, 2, 3} == S{1, 2, 3}'}}
static_assert(S{1, 2, 3} == S{~~1, 2, 3}); // expected-error {{failed}}

} // namespace

#ifdef TEST_CLIP
#define NO_LIMIT 'n'
#define SMALL 's'

namespace clipping_large_values {
  constexpr unsigned _BitInt(__BITINT_MAXWIDTH__ >> 12) Z = ~0;

#if TEST_CLIP == NO_LIMIT
  // expected-note@+6 {{'32317006071311007300714876688669951960444102669715484032130345427524655138867890893197201411522913463688717960921898019494119559150490921095088152386448283120630877367300996091750197750389652106796057638384067568276792218642619756161838094338476170470581645852036305042887575891541065808607552399123930385521914333389668342420684974786564569494856176035326322058077805659331026192708460314150258592864177116725943603718461857357598351152301645904403697613233287231227125684710820209725157101726931323469678542580656697935045997268352998638215525166389437335543602135433229604645318478604952148193555853611059596230655 == 1'}}
#elif TEST_CLIP == SMALL // fixme: see https://github.com/llvm/llvm-project/issues/71675
  // expected-note@+4 {{'32317006071311007300714876688669951960444102669715484032130345427524655138867890893197201411522913463688717960921898019494119559150490921095088152386448283120630877367300996091750197750389652106796057638384067568276792218642619756161838094338476170470581645852036305042887575891541065808607552399123930385521914333389668342420684974786564569494856176035326322058077805659331026192708460314150258592864177116725943603718461857357598351152301645904403697613233287231227125684710820209725157101726931323469678542580656697935045997268352998638215525166389437335543602135433229604645318478604952148193555853611059596230655 == 1'}}
#else // fixme: as above
  // expected-note@+2 {{'32317006071311007300714876688669951960444102669715484032130345427524655138867890893197201411522913463688717960921898019494119559150490921095088152386448283120630877367300996091750197750389652106796057638384067568276792218642619756161838094338476170470581645852036305042887575891541065808607552399123930385521914333389668342420684974786564569494856176035326322058077805659331026192708460314150258592864177116725943603718461857357598351152301645904403697613233287231227125684710820209725157101726931323469678542580656697935045997268352998638215525166389437335543602135433229604645318478604952148193555853611059596230655 == 1'}}
#endif
  static_assert(Z == 1); // expected-error {{failed}}

  constexpr struct {
    unsigned _BitInt(__BITINT_MAXWIDTH__ >> 12) F;

    constexpr bool operator==(const unsigned int& v) const {
      return F == v;
    }
  } my_god_its_full_of_bits = {(decltype(my_god_its_full_of_bits.F))~0};
  static_assert((decltype(my_god_its_full_of_bits)){} == 0);

#if TEST_CLIP == NO_LIMIT
  // expected-note@+6 {{'(decltype(clipping_large_values::my_god_its_full_of_bits)){32317006071311007300714876688669951960444102669715484032130345427524655138867890893197201411522913463688717960921898019494119559150490921095088152386448283120630877367300996091750197750389652106796057638384067568276792218642619756161838094338476170470581645852036305042887575891541065808607552399123930385521914333389668342420684974786564569494856176035326322058077805659331026192708460314150258592864177116725943603718461857357598351152301645904403697613233287231227125684710820209725157101726931323469678542580656697935045997268352998638215525166389437335543602135433229604645318478604952148193555853611059596230655} == 1'}}
#elif TEST_CLIP == SMALL
  // expected-note@+4 {{'(decltype(clipping_large_values::my_god_its_full_of_bits)){32317006071311007300714876688 ...(+559 bytes)... 52148193555853611059596230655} == 1'}}
#else
  // expected-note@+2 {{'(decltype(clipping_large_values::my_god_its_full_of_bits)){323170060713110073007148766886699519604441026697154840321303454275246551388678908931972014115229134636887179609218980194941195591504909210950881523864482831206 ...(+299 bytes)... 287231227125684710820209725157101726931323469678542580656697935045997268352998638215525166389437335543602135433229604645318478604952148193555853611059596230655} == 1'}}
#endif
  static_assert(my_god_its_full_of_bits == 1); // expected-error {{failed}}
}
#endif
