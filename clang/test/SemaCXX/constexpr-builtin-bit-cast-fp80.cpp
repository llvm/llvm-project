// RUN: %clang_cc1 -verify -std=c++2a -fsyntax-only -triple i386-pc-linux-gnu %s

// This is separate from constexpr-builtin-bit-cast.cpp because we want to
// compile for i386 so that sizeof(long double) is 12.

typedef long double fp80x2_v __attribute__((ext_vector_type(2)));

static_assert(sizeof(long double) == 12, "");
static_assert(sizeof(fp80x2_v) == 32, "");

struct fp80x2_s {
  char _data[2 * 10];
  unsigned char _pad[sizeof(fp80x2_v) - 2 * 10];

  constexpr bool operator==(const fp80x2_s& rhs) const {
    for (int i = 0; i < 2 * 10; ++i)
      if (_data[i] != rhs._data[i])
        return false;
    return true;
  }
};

constexpr static fp80x2_v test_vec_fp80 = { 1, 2 };
constexpr static fp80x2_s test_str_fp80 = { { 0, 0, 0, 0, 0, 0, 0, -128, -1, 63, 0, 0, 0, 0, 0, 0, 0, -128, 0, 64 }, {} };

// expected-error@+2 {{static assertion expression is not an integral constant expression}}
// expected-note@+1 {{constexpr bit_cast involving type 'long double' is not yet supported}}
static_assert(__builtin_bit_cast(fp80x2_s, test_vec_fp80) == test_str_fp80, "");

// expected-error@+2 {{static assertion expression is not an integral constant expression}}
// expected-note@+1 {{constexpr bit_cast involving type 'long double' is not yet supported}}
static_assert(__builtin_bit_cast(fp80x2_s, __builtin_bit_cast(fp80x2_v, test_str_fp80)) == test_str_fp80, "");

// expected-error@+2 {{constexpr variable 'bad_str_fp80_0' must be initialized by a constant expression}}
// expected-note@+1 {{constexpr bit_cast involving type 'long double' is not yet supported}}
constexpr static char bad_str_fp80_0 = __builtin_bit_cast(fp80x2_s, test_vec_fp80)._pad[0];

// expected-error@+2 {{constexpr variable 'bad_str_fp80_1' must be initialized by a constant expression}}
// expected-note@+1 {{constexpr bit_cast involving type 'long double' is not yet supported}}
constexpr static char bad_str_fp80_1 = __builtin_bit_cast(fp80x2_s, test_vec_fp80)._pad[1];

// expected-error@+2 {{constexpr variable 'bad_str_fp80_11' must be initialized by a constant expression}}
// expected-note@+1 {{constexpr bit_cast involving type 'long double' is not yet supported}}
constexpr static char bad_str_fp80_11 = __builtin_bit_cast(fp80x2_s, test_vec_fp80)._pad[11];

// expected-error@+2 {{constexpr variable 'struct2v' must be initialized by a constant expression}}
// expected-note@+1 {{constexpr bit_cast involving type 'long double' is not yet supported}}
constexpr static fp80x2_v struct2v = __builtin_bit_cast(fp80x2_v, test_str_fp80);
