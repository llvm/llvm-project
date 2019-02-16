// RUN: %clang_cc1 -std=c++11 -triple i386-apple-darwin9 -fsyntax-only -verify %s
// expected-no-diagnostics

using size_t = decltype(sizeof(0));

struct complex_double {
  double real;
  double imag;
};

template <typename T, size_t ABI, size_t Preferred>
struct check_alignment {
  using type = T;
  static type value;

  static_assert(__alignof__(value) == Preferred, "__alignof__(value) != Preferred");
  static_assert(__alignof__(type) == Preferred, "__alignof__(type) != Preferred");
  static_assert(alignof(type) == ABI, "alignof(type) != ABI");
};

// PR3433
template struct check_alignment<double, 4, 8>;
template struct check_alignment<long long, 4, 8>;
template struct check_alignment<unsigned long long, 4, 8>;
template struct check_alignment<complex_double, 4, 4>;

// PR6362
struct __attribute__((packed))
packed_struct {
  unsigned int a;
} g_packedstruct;
template struct check_alignment<packed_struct, 1, 1>;
static_assert(__alignof__(g_packedstruct.a) == 1, "__alignof__(packed_struct.member) != 1");

template struct check_alignment<double[3], 4, 8>;

enum big_enum { x = 18446744073709551615ULL };
template struct check_alignment<big_enum, 4, 8>;

// PR5637

#define ALIGNED(x) __attribute__((aligned(x)))

typedef ALIGNED(2) struct {
  char a[3];
} aligned_before_struct;

static_assert(sizeof(aligned_before_struct)       == 3, "");
static_assert(sizeof(aligned_before_struct[1])    == 4, "");
static_assert(sizeof(aligned_before_struct[2])    == 6, "");
static_assert(sizeof(aligned_before_struct[2][1]) == 8, "");
static_assert(sizeof(aligned_before_struct[1][2]) == 6, "");

typedef struct ALIGNED(2) {
  char a[3];
} aligned_after_struct;

static_assert(sizeof(aligned_after_struct)       == 4, "");
static_assert(sizeof(aligned_after_struct[1])    == 4, "");
static_assert(sizeof(aligned_after_struct[2])    == 8, "");
static_assert(sizeof(aligned_after_struct[2][1]) == 8, "");
static_assert(sizeof(aligned_after_struct[1][2]) == 8, "");
