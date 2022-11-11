// RUN: %clang_cc1 -std=c11 -triple i386-apple-darwin9 -fsyntax-only -verify %s

#define STATIC_ASSERT(cond) _Static_assert(cond, #cond)

// PR3433
#define CHECK_ALIGNMENT(type, name, abi, pref) \
  type name; \
  STATIC_ASSERT(__alignof__(name) == pref); \
  STATIC_ASSERT(__alignof__(type) == pref); \
  STATIC_ASSERT(_Alignof(type) == abi)

CHECK_ALIGNMENT(double, g_double, 4, 8);
CHECK_ALIGNMENT(long long, g_longlong, 4, 8);
CHECK_ALIGNMENT(unsigned long long, g_ulonglong, 4, 8);
CHECK_ALIGNMENT(_Complex double, g_complexdouble, 4, 8);

// PR6362
struct __attribute__((packed))
packed_struct {
  unsigned int a;
};
CHECK_ALIGNMENT(struct packed_struct, g_packedstruct, 1, 1);
STATIC_ASSERT(__alignof__(g_packedstruct.a) == 1);

typedef double arr3double[3];
CHECK_ALIGNMENT(arr3double, g_arr3double, 4, 8);

enum big_enum { x = 18446744073709551615ULL };
CHECK_ALIGNMENT(enum big_enum, g_bigenum, 4, 8);

// PR5637

#define ALIGNED(x) __attribute__((aligned(x)))

typedef ALIGNED(2) struct {
  char a[3];
} aligned_before_struct;

STATIC_ASSERT(sizeof(aligned_before_struct)       == 3);
STATIC_ASSERT(sizeof(aligned_before_struct[1])    == 4); // expected-error {{size of array element}}
STATIC_ASSERT(sizeof(aligned_before_struct[2])    == 6); // expected-error {{size of array element}}
STATIC_ASSERT(sizeof(aligned_before_struct[2][1]) == 8); // expected-error {{size of array element}}
STATIC_ASSERT(sizeof(aligned_before_struct[1][2]) == 6); // expected-error {{size of array element}}

typedef struct ALIGNED(2) {
  char a[3];
} aligned_after_struct;

STATIC_ASSERT(sizeof(aligned_after_struct)       == 4);
STATIC_ASSERT(sizeof(aligned_after_struct[1])    == 4);
STATIC_ASSERT(sizeof(aligned_after_struct[2])    == 8);
STATIC_ASSERT(sizeof(aligned_after_struct[2][1]) == 8);
STATIC_ASSERT(sizeof(aligned_after_struct[1][2]) == 8);
