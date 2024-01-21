// RUN: %clang_cc1 -verify -std=c++2a -fsyntax-only %s
// expected-no-diagnostics

#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
#define LITTLE_END 1
#elif __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#define LITTLE_END 0
#else
#error "huh?"
#endif

typedef double vector4double __attribute__((__vector_size__(32)));
typedef float vector4float __attribute__((__vector_size__(16)));
typedef long long vector4long __attribute__((__vector_size__(32)));
typedef int vector4int __attribute__((__vector_size__(16)));
typedef short vector4short __attribute__((__vector_size__(8)));
typedef char vector4char __attribute__((__vector_size__(4)));
typedef double vector8double __attribute__((__vector_size__(64)));
typedef float vector8float __attribute__((__vector_size__(32)));
typedef long long vector8long __attribute__((__vector_size__(64)));
typedef int vector8int __attribute__((__vector_size__(32)));
typedef short vector8short __attribute__((__vector_size__(16)));
typedef char vector8char __attribute__((__vector_size__(8)));

#define CHECK_NUM(__size, __typeFrom, __typeTo, ...)                            \
  constexpr vector##__size##__typeTo                                            \
      from_##vector##__size##__typeFrom##_to_##vector##__size##__typeTo##_var = \
          __builtin_convertvector((vector##__size##__typeFrom){__VA_ARGS__},    \
                                  vector##__size##__typeTo);
#define CHECK_TO_ALL_TYPES(__size, __typeFrom, ...)                            \
  CHECK_NUM(__size, __typeFrom, double, __VA_ARGS__)                           \
  CHECK_NUM(__size, __typeFrom, float, __VA_ARGS__)                            \
  CHECK_NUM(__size, __typeFrom, long, __VA_ARGS__)                             \
  CHECK_NUM(__size, __typeFrom, int, __VA_ARGS__)                              \
  CHECK_NUM(__size, __typeFrom, short, __VA_ARGS__)                            \
  CHECK_NUM(__size, __typeFrom, char, __VA_ARGS__)                             \
  static_assert(                                                               \
      __builtin_bit_cast(                                                      \
          unsigned,                                                            \
          __builtin_shufflevector(                                             \
              from_vector##__size##__typeFrom##_to_vector##__size##char_var,   \
              from_vector##__size##__typeFrom##_to_vector##__size##char_var,   \
              0, 1, 2, 3)) == (LITTLE_END ? 0x03020100 : 0x00010203));         \
  static_assert(                                                               \
      __builtin_bit_cast(                                                      \
          unsigned long long,                                                  \
          __builtin_shufflevector(                                             \
              from_vector##__size##__typeFrom##_to_vector##__size##short_var,  \
              from_vector##__size##__typeFrom##_to_vector##__size##short_var,  \
              0, 1, 2, 3)) ==                                                  \
      (LITTLE_END ? 0x0003000200010000 : 0x0000000100020003));

#define CHECK_ALL_COMBINATIONS(__size, ...)                                    \
  CHECK_TO_ALL_TYPES(__size, double, __VA_ARGS__)                              \
  CHECK_TO_ALL_TYPES(__size, float, __VA_ARGS__)                               \
  CHECK_TO_ALL_TYPES(__size, long, __VA_ARGS__)                                \
  CHECK_TO_ALL_TYPES(__size, int, __VA_ARGS__)                                 \
  CHECK_TO_ALL_TYPES(__size, short, __VA_ARGS__)                               \
  CHECK_TO_ALL_TYPES(__size, char, __VA_ARGS__)

CHECK_ALL_COMBINATIONS(4, 0, 1, 2, 3);
CHECK_ALL_COMBINATIONS(8, 0, 1, 2, 3, 4, 5, 6, 7);
#undef CHECK_ALL_COMBINATIONS
#undef CHECK_TO_ALL_TYPES
#undef CHECK_NUM

// Shuffle vector
constexpr vector4char vector4charConst1 = {0, 1, 2, 3};
constexpr vector4char vector4charConst2 = {4, 5, 6, 7};
constexpr vector8char vector8intConst = {8, 9, 10, 11, 12, 13, 14, 15};

constexpr vector4char vectorShuffle1 =
    __builtin_shufflevector(vector4charConst1, vector4charConst2, 0, 1, 2, 3);
static_assert(__builtin_bit_cast(unsigned, vectorShuffle1) ==
              (LITTLE_END ? 0x03020100 : 0x00010203));
constexpr vector4char vectorShuffle2 =
    __builtin_shufflevector(vector4charConst1, vector4charConst2, 4, 5, 6, 7);
static_assert(__builtin_bit_cast(unsigned, vectorShuffle2) ==
              (LITTLE_END ? 0x07060504 : 0x04050607));
constexpr vector4char vectorShuffle3 = __builtin_shufflevector(
    vector4charConst1, vector4charConst2, -1, -1, -1, -1);
static_assert(__builtin_bit_cast(unsigned, vectorShuffle3) ==
              (LITTLE_END ? 0x00000000 : 0x00000000));
constexpr vector4char vectorShuffle4 =
    __builtin_shufflevector(vector4charConst1, vector4charConst2, 0, 2, 4, 6);
static_assert(__builtin_bit_cast(unsigned, vectorShuffle4) ==
              (LITTLE_END ? 0x06040200 : 0x00020406));
constexpr vector8char vectorShuffle5 = __builtin_shufflevector(
    vector8intConst, vector8intConst, 0, 2, 4, 6, 8, 10, 12, 14);
static_assert(__builtin_bit_cast(unsigned long long, vectorShuffle5) ==
              (LITTLE_END ? 0x0E0C0A080E0C0A08 : 0x080A0C0E080A0C0E));
constexpr vector4char vectorShuffle6 =
    __builtin_shufflevector(vector8intConst, vector8intConst, 0, 2, 4, 6);
static_assert(__builtin_bit_cast(unsigned, vectorShuffle6) ==
              (LITTLE_END ? 0x0E0C0A08 : 0x080A0C0E));
constexpr vector8char vectorShuffle7 = __builtin_shufflevector(
    vector4charConst1, vector4charConst2, 0, 2, 4, 6, 1, 3, 5, 7);
static_assert(__builtin_bit_cast(unsigned long long, vectorShuffle7) ==
              (LITTLE_END ? 0x0705030106040200 : 0x0002040601030507));
