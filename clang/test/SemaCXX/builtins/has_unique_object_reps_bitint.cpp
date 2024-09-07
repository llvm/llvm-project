// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fsyntax-only -verify -std=c++17 -Wno-bitfield-width %s
//  expected-no-diagnostics

static_assert(__has_unique_object_representations(_BitInt(8)));
static_assert(__has_unique_object_representations(unsigned _BitInt(8)));
static_assert(__has_unique_object_representations(_BitInt(sizeof(int) * 8u)));
// sizeof(_BitInt(24)) may be 4 to align it to the next greater integer type, in which case it would have 8 padding bits.
static_assert(__has_unique_object_representations(_BitInt(24)) == (sizeof(_BitInt(24)) == 3));

static_assert(!__has_unique_object_representations(_BitInt(7)));
static_assert(!__has_unique_object_representations(unsigned _BitInt(7)));
static_assert(!__has_unique_object_representations(_BitInt(2)));
static_assert(!__has_unique_object_representations(unsigned _BitInt(1)));

template <unsigned N>
constexpr bool check() {
  if constexpr (N <= __BITINT_MAXWIDTH__) {
    static_assert(__has_unique_object_representations(_BitInt(N)) == (sizeof(_BitInt(N)) * 8u == N));
    static_assert(__has_unique_object_representations(unsigned _BitInt(N)) == (sizeof(unsigned _BitInt(N)) * 8u == N));
  }
  return true;
}

template <unsigned... N>
constexpr bool do_check = (check<N>() && ...);

static_assert(do_check<2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18>);
static_assert(do_check<15, 16, 17, 23, 24, 25, 31, 32, 33>);
static_assert(do_check<39, 40, 41, 47, 48, 49>);
static_assert(do_check<127, 128, 129, 255, 256, 257, 383, 384, 385>);

template <unsigned N>
struct in_struct {
  _BitInt(N) x;
  static constexpr bool check() {
    return __has_unique_object_representations(in_struct<N>) == __has_unique_object_representations(_BitInt(N));
  }
};

static_assert(in_struct<8>::check());
static_assert(in_struct<7>::check());

struct bit_fields_1 {
  _BitInt(7) x : 7;
  unsigned _BitInt(1) y : 1;
};

static_assert(__has_unique_object_representations(bit_fields_1) == (sizeof(bit_fields_1) == 1));

struct bit_fields_2 {
  _BitInt(8) x : 7;
};

static_assert(!__has_unique_object_representations(bit_fields_2));

struct bit_fields_3 {
  _BitInt(15) x : 8;
};

static_assert(__has_unique_object_representations(bit_fields_3) == (sizeof(bit_fields_3) == 1));

#if __BITINT_MAXWIDTH__ >= 129
struct bit_fields_4 {
  _BitInt(129) x : 128;
};

static_assert(__has_unique_object_representations(bit_fields_4) == (sizeof(bit_fields_4) == 128 / 8));
#endif

struct bit_fields_5 {
  _BitInt(2) x : 8;
};

static_assert(!__has_unique_object_representations(bit_fields_5));

template <unsigned N>
struct ref_member {
  _BitInt(N) & x;
};

struct int_ref_member {
  int &x;
};

static_assert(__has_unique_object_representations(ref_member<7>) == __has_unique_object_representations(ref_member<8>));
static_assert(__has_unique_object_representations(ref_member<8>) == __has_unique_object_representations(int_ref_member));
