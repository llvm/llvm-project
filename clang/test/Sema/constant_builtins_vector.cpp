// RUN: %clang_cc1 -verify -std=c++2a -fsyntax-only -Wno-bit-int-extension %s
// RUN: %clang_cc1 -verify -std=c++2a -fsyntax-only -Wno-bit-int-extension -triple ppc64-unknown-linux %s
// RUN: %clang_cc1 -verify -std=c++2a -fsyntax-only -Wno-bit-int-extension -triple ppc64le-unknown-linux %s

#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
#define LITTLE_END 1
#elif __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#define LITTLE_END 0
#else
#error "huh?"
#endif

// We also support _BitInt as long as it is >=8 and a power of 2.
typedef _BitInt(8) BitInt8;
typedef _BitInt(32) BitInt32;
typedef _BitInt(128) BitInt128;

typedef double vector4double __attribute__((__vector_size__(32)));
typedef float vector4float __attribute__((__vector_size__(16)));
typedef long long vector4long __attribute__((__vector_size__(32)));
typedef int vector4int __attribute__((__vector_size__(16)));
typedef unsigned long long vector4ulong __attribute__((__vector_size__(32)));
typedef unsigned int vector4uint __attribute__((__vector_size__(16)));
typedef short vector4short __attribute__((__vector_size__(8)));
typedef char vector4char __attribute__((__vector_size__(4)));
typedef BitInt8 vector4BitInt8 __attribute__((__vector_size__(4)));
typedef BitInt32 vector4BitInt32 __attribute__((__vector_size__(16)));
typedef BitInt128 vector4BitInt128 __attribute__((__vector_size__(64)));
typedef double vector8double __attribute__((__vector_size__(64)));
typedef float vector8float __attribute__((__vector_size__(32)));
typedef long long vector8long __attribute__((__vector_size__(64)));
typedef int vector8int __attribute__((__vector_size__(32)));
typedef short vector8short __attribute__((__vector_size__(16)));
typedef char vector8char __attribute__((__vector_size__(8)));
typedef BitInt8 vector8BitInt8 __attribute__((__vector_size__(8)));
typedef BitInt32 vector8BitInt32 __attribute__((__vector_size__(32)));
typedef BitInt128 vector8BitInt128 __attribute__((__vector_size__(128)));

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
  CHECK_NUM(__size, __typeFrom, BitInt8, __VA_ARGS__)                          \
  CHECK_NUM(__size, __typeFrom, BitInt32, __VA_ARGS__)                         \
  CHECK_NUM(__size, __typeFrom, BitInt128, __VA_ARGS__)                        \
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
  CHECK_TO_ALL_TYPES(__size, char, __VA_ARGS__)                                \
  CHECK_TO_ALL_TYPES(__size, BitInt8, __VA_ARGS__)                             \
  CHECK_TO_ALL_TYPES(__size, BitInt32, __VA_ARGS__)                            \
  CHECK_TO_ALL_TYPES(__size, BitInt128, __VA_ARGS__)

// The result below is expanded from these macros. Use them to autogenerate the
// test cases below.
// CHECK_ALL_COMBINATIONS(4, 0, 1, 2, 3);
// CHECK_ALL_COMBINATIONS(8, 0, 1, 2, 3, 4, 5, 6, 7);

constexpr vector4double from_vector4double_to_vector4double_var =
    __builtin_convertvector((vector4double){0, 1, 2, 3}, vector4double);
constexpr vector4float from_vector4double_to_vector4float_var =
    __builtin_convertvector((vector4double){0, 1, 2, 3}, vector4float);
constexpr vector4long from_vector4double_to_vector4long_var =
    __builtin_convertvector((vector4double){0, 1, 2, 3}, vector4long);
constexpr vector4int from_vector4double_to_vector4int_var =
    __builtin_convertvector((vector4double){0, 1, 2, 3}, vector4int);
constexpr vector4short from_vector4double_to_vector4short_var =
    __builtin_convertvector((vector4double){0, 1, 2, 3}, vector4short);
constexpr vector4char from_vector4double_to_vector4char_var =
    __builtin_convertvector((vector4double){0, 1, 2, 3}, vector4char);
constexpr vector4BitInt8 from_vector4double_to_vector4BitInt8_var =
    __builtin_convertvector((vector4double){0, 1, 2, 3}, vector4BitInt8);
constexpr vector4BitInt32 from_vector4double_to_vector4BitInt32_var =
    __builtin_convertvector((vector4double){0, 1, 2, 3}, vector4BitInt32);
constexpr vector4BitInt128 from_vector4double_to_vector4BitInt128_var =
    __builtin_convertvector((vector4double){0, 1, 2, 3}, vector4BitInt128);
static_assert(__builtin_bit_cast(
                  unsigned,
                  __builtin_shufflevector(from_vector4double_to_vector4char_var,
                                          from_vector4double_to_vector4char_var,
                                          0, 1, 2, 3)) ==
              (LITTLE_END ? 0x03020100 : 0x00010203));
static_assert(__builtin_bit_cast(unsigned long long,
                                 __builtin_shufflevector(
                                     from_vector4double_to_vector4short_var,
                                     from_vector4double_to_vector4short_var, 0,
                                     1, 2, 3)) ==
              (LITTLE_END ? 0x0003000200010000 : 0x0000000100020003));
constexpr vector4double from_vector4float_to_vector4double_var =
    __builtin_convertvector((vector4float){0, 1, 2, 3}, vector4double);
constexpr vector4float from_vector4float_to_vector4float_var =
    __builtin_convertvector((vector4float){0, 1, 2, 3}, vector4float);
constexpr vector4long from_vector4float_to_vector4long_var =
    __builtin_convertvector((vector4float){0, 1, 2, 3}, vector4long);
constexpr vector4int from_vector4float_to_vector4int_var =
    __builtin_convertvector((vector4float){0, 1, 2, 3}, vector4int);
constexpr vector4short from_vector4float_to_vector4short_var =
    __builtin_convertvector((vector4float){0, 1, 2, 3}, vector4short);
constexpr vector4char from_vector4float_to_vector4char_var =
    __builtin_convertvector((vector4float){0, 1, 2, 3}, vector4char);
constexpr vector4BitInt8 from_vector4float_to_vector4BitInt8_var =
    __builtin_convertvector((vector4float){0, 1, 2, 3}, vector4BitInt8);
constexpr vector4BitInt32 from_vector4float_to_vector4BitInt32_var =
    __builtin_convertvector((vector4float){0, 1, 2, 3}, vector4BitInt32);
constexpr vector4BitInt128 from_vector4float_to_vector4BitInt128_var =
    __builtin_convertvector((vector4float){0, 1, 2, 3}, vector4BitInt128);
static_assert(__builtin_bit_cast(unsigned,
                                 __builtin_shufflevector(
                                     from_vector4float_to_vector4char_var,
                                     from_vector4float_to_vector4char_var, 0, 1,
                                     2, 3)) == (LITTLE_END ? 0x03020100 : 0x00010203));
static_assert(__builtin_bit_cast(
                  unsigned long long,
                  __builtin_shufflevector(from_vector4float_to_vector4short_var,
                                          from_vector4float_to_vector4short_var,
                                          0, 1, 2, 3)) ==
              (LITTLE_END ? 0x0003000200010000 : 0x0000000100020003));
constexpr vector4double from_vector4long_to_vector4double_var =
    __builtin_convertvector((vector4long){0, 1, 2, 3}, vector4double);
constexpr vector4float from_vector4long_to_vector4float_var =
    __builtin_convertvector((vector4long){0, 1, 2, 3}, vector4float);
constexpr vector4long from_vector4long_to_vector4long_var =
    __builtin_convertvector((vector4long){0, 1, 2, 3}, vector4long);
constexpr vector4int from_vector4long_to_vector4int_var =
    __builtin_convertvector((vector4long){0, 1, 2, 3}, vector4int);
constexpr vector4short from_vector4long_to_vector4short_var =
    __builtin_convertvector((vector4long){0, 1, 2, 3}, vector4short);
constexpr vector4char from_vector4long_to_vector4char_var =
    __builtin_convertvector((vector4long){0, 1, 2, 3}, vector4char);
constexpr vector4BitInt8 from_vector4long_to_vector4BitInt8_var =
    __builtin_convertvector((vector4long){0, 1, 2, 3}, vector4BitInt8);
constexpr vector4BitInt32 from_vector4long_to_vector4BitInt32_var =
    __builtin_convertvector((vector4long){0, 1, 2, 3}, vector4BitInt32);
constexpr vector4BitInt128 from_vector4long_to_vector4BitInt128_var =
    __builtin_convertvector((vector4long){0, 1, 2, 3}, vector4BitInt128);
static_assert(__builtin_bit_cast(unsigned,
                                 __builtin_shufflevector(
                                     from_vector4long_to_vector4char_var,
                                     from_vector4long_to_vector4char_var, 0, 1,
                                     2, 3)) == (LITTLE_END ? 0x03020100 : 0x00010203));
static_assert(__builtin_bit_cast(
                  unsigned long long,
                  __builtin_shufflevector(from_vector4long_to_vector4short_var,
                                          from_vector4long_to_vector4short_var,
                                          0, 1, 2, 3)) ==
              (LITTLE_END ? 0x0003000200010000 : 0x0000000100020003));
constexpr vector4double from_vector4int_to_vector4double_var =
    __builtin_convertvector((vector4int){0, 1, 2, 3}, vector4double);
constexpr vector4float from_vector4int_to_vector4float_var =
    __builtin_convertvector((vector4int){0, 1, 2, 3}, vector4float);
constexpr vector4long from_vector4int_to_vector4long_var =
    __builtin_convertvector((vector4int){0, 1, 2, 3}, vector4long);
constexpr vector4int from_vector4int_to_vector4int_var =
    __builtin_convertvector((vector4int){0, 1, 2, 3}, vector4int);
constexpr vector4short from_vector4int_to_vector4short_var =
    __builtin_convertvector((vector4int){0, 1, 2, 3}, vector4short);
constexpr vector4char from_vector4int_to_vector4char_var =
    __builtin_convertvector((vector4int){0, 1, 2, 3}, vector4char);
constexpr vector4BitInt8 from_vector4int_to_vector4BitInt8_var =
    __builtin_convertvector((vector4int){0, 1, 2, 3}, vector4BitInt8);
constexpr vector4BitInt32 from_vector4int_to_vector4BitInt32_var =
    __builtin_convertvector((vector4int){0, 1, 2, 3}, vector4BitInt32);
constexpr vector4BitInt128 from_vector4int_to_vector4BitInt128_var =
    __builtin_convertvector((vector4int){0, 1, 2, 3}, vector4BitInt128);
static_assert(__builtin_bit_cast(unsigned,
                                 __builtin_shufflevector(
                                     from_vector4int_to_vector4char_var,
                                     from_vector4int_to_vector4char_var, 0, 1,
                                     2, 3)) == (LITTLE_END ? 0x03020100 : 0x00010203));
static_assert(__builtin_bit_cast(
                  unsigned long long,
                  __builtin_shufflevector(from_vector4int_to_vector4short_var,
                                          from_vector4int_to_vector4short_var,
                                          0, 1, 2, 3)) ==
              (LITTLE_END ? 0x0003000200010000 : 0x0000000100020003));
constexpr vector4double from_vector4short_to_vector4double_var =
    __builtin_convertvector((vector4short){0, 1, 2, 3}, vector4double);
constexpr vector4float from_vector4short_to_vector4float_var =
    __builtin_convertvector((vector4short){0, 1, 2, 3}, vector4float);
constexpr vector4long from_vector4short_to_vector4long_var =
    __builtin_convertvector((vector4short){0, 1, 2, 3}, vector4long);
constexpr vector4int from_vector4short_to_vector4int_var =
    __builtin_convertvector((vector4short){0, 1, 2, 3}, vector4int);
constexpr vector4short from_vector4short_to_vector4short_var =
    __builtin_convertvector((vector4short){0, 1, 2, 3}, vector4short);
constexpr vector4char from_vector4short_to_vector4char_var =
    __builtin_convertvector((vector4short){0, 1, 2, 3}, vector4char);
constexpr vector4BitInt8 from_vector4short_to_vector4BitInt8_var =
    __builtin_convertvector((vector4short){0, 1, 2, 3}, vector4BitInt8);
constexpr vector4BitInt32 from_vector4short_to_vector4BitInt32_var =
    __builtin_convertvector((vector4short){0, 1, 2, 3}, vector4BitInt32);
constexpr vector4BitInt128 from_vector4short_to_vector4BitInt128_var =
    __builtin_convertvector((vector4short){0, 1, 2, 3}, vector4BitInt128);
static_assert(__builtin_bit_cast(unsigned,
                                 __builtin_shufflevector(
                                     from_vector4short_to_vector4char_var,
                                     from_vector4short_to_vector4char_var, 0, 1,
                                     2, 3)) == (LITTLE_END ? 0x03020100 : 0x00010203));
static_assert(__builtin_bit_cast(
                  unsigned long long,
                  __builtin_shufflevector(from_vector4short_to_vector4short_var,
                                          from_vector4short_to_vector4short_var,
                                          0, 1, 2, 3)) ==
              (LITTLE_END ? 0x0003000200010000 : 0x0000000100020003));
constexpr vector4double from_vector4char_to_vector4double_var =
    __builtin_convertvector((vector4char){0, 1, 2, 3}, vector4double);
constexpr vector4float from_vector4char_to_vector4float_var =
    __builtin_convertvector((vector4char){0, 1, 2, 3}, vector4float);
constexpr vector4long from_vector4char_to_vector4long_var =
    __builtin_convertvector((vector4char){0, 1, 2, 3}, vector4long);
constexpr vector4int from_vector4char_to_vector4int_var =
    __builtin_convertvector((vector4char){0, 1, 2, 3}, vector4int);
constexpr vector4short from_vector4char_to_vector4short_var =
    __builtin_convertvector((vector4char){0, 1, 2, 3}, vector4short);
constexpr vector4char from_vector4char_to_vector4char_var =
    __builtin_convertvector((vector4char){0, 1, 2, 3}, vector4char);
constexpr vector4BitInt8 from_vector4char_to_vector4BitInt8_var =
    __builtin_convertvector((vector4char){0, 1, 2, 3}, vector4BitInt8);
constexpr vector4BitInt32 from_vector4char_to_vector4BitInt32_var =
    __builtin_convertvector((vector4char){0, 1, 2, 3}, vector4BitInt32);
constexpr vector4BitInt128 from_vector4char_to_vector4BitInt128_var =
    __builtin_convertvector((vector4char){0, 1, 2, 3}, vector4BitInt128);
static_assert(__builtin_bit_cast(unsigned,
                                 __builtin_shufflevector(
                                     from_vector4char_to_vector4char_var,
                                     from_vector4char_to_vector4char_var, 0, 1,
                                     2, 3)) == (LITTLE_END ? 0x03020100 : 0x00010203));
static_assert(__builtin_bit_cast(
                  unsigned long long,
                  __builtin_shufflevector(from_vector4char_to_vector4short_var,
                                          from_vector4char_to_vector4short_var,
                                          0, 1, 2, 3)) ==
              (LITTLE_END ? 0x0003000200010000 : 0x0000000100020003));
constexpr vector4double from_vector4BitInt8_to_vector4double_var =
    __builtin_convertvector((vector4BitInt8){0, 1, 2, 3}, vector4double);
constexpr vector4float from_vector4BitInt8_to_vector4float_var =
    __builtin_convertvector((vector4BitInt8){0, 1, 2, 3}, vector4float);
constexpr vector4long from_vector4BitInt8_to_vector4long_var =
    __builtin_convertvector((vector4BitInt8){0, 1, 2, 3}, vector4long);
constexpr vector4int from_vector4BitInt8_to_vector4int_var =
    __builtin_convertvector((vector4BitInt8){0, 1, 2, 3}, vector4int);
constexpr vector4short from_vector4BitInt8_to_vector4short_var =
    __builtin_convertvector((vector4BitInt8){0, 1, 2, 3}, vector4short);
constexpr vector4char from_vector4BitInt8_to_vector4char_var =
    __builtin_convertvector((vector4BitInt8){0, 1, 2, 3}, vector4char);
constexpr vector4BitInt8 from_vector4BitInt8_to_vector4BitInt8_var =
    __builtin_convertvector((vector4BitInt8){0, 1, 2, 3}, vector4BitInt8);
constexpr vector4BitInt32 from_vector4BitInt8_to_vector4BitInt32_var =
    __builtin_convertvector((vector4BitInt8){0, 1, 2, 3}, vector4BitInt32);
constexpr vector4BitInt128 from_vector4BitInt8_to_vector4BitInt128_var =
    __builtin_convertvector((vector4BitInt8){0, 1, 2, 3}, vector4BitInt128);
static_assert(__builtin_bit_cast(unsigned,
                                 __builtin_shufflevector(
                                     from_vector4BitInt8_to_vector4char_var,
                                     from_vector4BitInt8_to_vector4char_var, 0,
                                     1, 2, 3)) ==
              (LITTLE_END ? 0x03020100 : 0x00010203));
static_assert(__builtin_bit_cast(unsigned long long,
                                 __builtin_shufflevector(
                                     from_vector4BitInt8_to_vector4short_var,
                                     from_vector4BitInt8_to_vector4short_var, 0,
                                     1, 2, 3)) ==
              (LITTLE_END ? 0x0003000200010000 : 0x0000000100020003));
constexpr vector4double from_vector4BitInt32_to_vector4double_var =
    __builtin_convertvector((vector4BitInt32){0, 1, 2, 3}, vector4double);
constexpr vector4float from_vector4BitInt32_to_vector4float_var =
    __builtin_convertvector((vector4BitInt32){0, 1, 2, 3}, vector4float);
constexpr vector4long from_vector4BitInt32_to_vector4long_var =
    __builtin_convertvector((vector4BitInt32){0, 1, 2, 3}, vector4long);
constexpr vector4int from_vector4BitInt32_to_vector4int_var =
    __builtin_convertvector((vector4BitInt32){0, 1, 2, 3}, vector4int);
constexpr vector4short from_vector4BitInt32_to_vector4short_var =
    __builtin_convertvector((vector4BitInt32){0, 1, 2, 3}, vector4short);
constexpr vector4char from_vector4BitInt32_to_vector4char_var =
    __builtin_convertvector((vector4BitInt32){0, 1, 2, 3}, vector4char);
constexpr vector4BitInt8 from_vector4BitInt32_to_vector4BitInt8_var =
    __builtin_convertvector((vector4BitInt32){0, 1, 2, 3}, vector4BitInt8);
constexpr vector4BitInt32 from_vector4BitInt32_to_vector4BitInt32_var =
    __builtin_convertvector((vector4BitInt32){0, 1, 2, 3}, vector4BitInt32);
constexpr vector4BitInt128 from_vector4BitInt32_to_vector4BitInt128_var =
    __builtin_convertvector((vector4BitInt32){0, 1, 2, 3}, vector4BitInt128);
static_assert(__builtin_bit_cast(unsigned,
                                 __builtin_shufflevector(
                                     from_vector4BitInt32_to_vector4char_var,
                                     from_vector4BitInt32_to_vector4char_var, 0,
                                     1, 2, 3)) ==
              (LITTLE_END ? 0x03020100 : 0x00010203));
static_assert(__builtin_bit_cast(unsigned long long,
                                 __builtin_shufflevector(
                                     from_vector4BitInt32_to_vector4short_var,
                                     from_vector4BitInt32_to_vector4short_var,
                                     0, 1, 2, 3)) ==
              (LITTLE_END ? 0x0003000200010000 : 0x0000000100020003));
constexpr vector4double from_vector4BitInt128_to_vector4double_var =
    __builtin_convertvector((vector4BitInt128){0, 1, 2, 3}, vector4double);
constexpr vector4float from_vector4BitInt128_to_vector4float_var =
    __builtin_convertvector((vector4BitInt128){0, 1, 2, 3}, vector4float);
constexpr vector4long from_vector4BitInt128_to_vector4long_var =
    __builtin_convertvector((vector4BitInt128){0, 1, 2, 3}, vector4long);
constexpr vector4int from_vector4BitInt128_to_vector4int_var =
    __builtin_convertvector((vector4BitInt128){0, 1, 2, 3}, vector4int);
constexpr vector4short from_vector4BitInt128_to_vector4short_var =
    __builtin_convertvector((vector4BitInt128){0, 1, 2, 3}, vector4short);
constexpr vector4char from_vector4BitInt128_to_vector4char_var =
    __builtin_convertvector((vector4BitInt128){0, 1, 2, 3}, vector4char);
constexpr vector4BitInt8 from_vector4BitInt128_to_vector4BitInt8_var =
    __builtin_convertvector((vector4BitInt128){0, 1, 2, 3}, vector4BitInt8);
constexpr vector4BitInt32 from_vector4BitInt128_to_vector4BitInt32_var =
    __builtin_convertvector((vector4BitInt128){0, 1, 2, 3}, vector4BitInt32);
constexpr vector4BitInt128 from_vector4BitInt128_to_vector4BitInt128_var =
    __builtin_convertvector((vector4BitInt128){0, 1, 2, 3}, vector4BitInt128);
static_assert(__builtin_bit_cast(unsigned,
                                 __builtin_shufflevector(
                                     from_vector4BitInt128_to_vector4char_var,
                                     from_vector4BitInt128_to_vector4char_var,
                                     0, 1, 2, 3)) ==
              (LITTLE_END ? 0x03020100 : 0x00010203));
static_assert(__builtin_bit_cast(unsigned long long,
                                 __builtin_shufflevector(
                                     from_vector4BitInt128_to_vector4short_var,
                                     from_vector4BitInt128_to_vector4short_var,
                                     0, 1, 2, 3)) ==
              (LITTLE_END ? 0x0003000200010000 : 0x0000000100020003));
;
constexpr vector8double from_vector8double_to_vector8double_var =
    __builtin_convertvector((vector8double){0, 1, 2, 3, 4, 5, 6, 7},
                            vector8double);
constexpr vector8float from_vector8double_to_vector8float_var =
    __builtin_convertvector((vector8double){0, 1, 2, 3, 4, 5, 6, 7},
                            vector8float);
constexpr vector8long from_vector8double_to_vector8long_var =
    __builtin_convertvector((vector8double){0, 1, 2, 3, 4, 5, 6, 7},
                            vector8long);
constexpr vector8int from_vector8double_to_vector8int_var =
    __builtin_convertvector((vector8double){0, 1, 2, 3, 4, 5, 6, 7},
                            vector8int);
constexpr vector8short from_vector8double_to_vector8short_var =
    __builtin_convertvector((vector8double){0, 1, 2, 3, 4, 5, 6, 7},
                            vector8short);
constexpr vector8char from_vector8double_to_vector8char_var =
    __builtin_convertvector((vector8double){0, 1, 2, 3, 4, 5, 6, 7},
                            vector8char);
constexpr vector8BitInt8 from_vector8double_to_vector8BitInt8_var =
    __builtin_convertvector((vector8double){0, 1, 2, 3, 4, 5, 6, 7},
                            vector8BitInt8);
constexpr vector8BitInt32 from_vector8double_to_vector8BitInt32_var =
    __builtin_convertvector((vector8double){0, 1, 2, 3, 4, 5, 6, 7},
                            vector8BitInt32);
constexpr vector8BitInt128 from_vector8double_to_vector8BitInt128_var =
    __builtin_convertvector((vector8double){0, 1, 2, 3, 4, 5, 6, 7},
                            vector8BitInt128);
static_assert(__builtin_bit_cast(
                  unsigned,
                  __builtin_shufflevector(from_vector8double_to_vector8char_var,
                                          from_vector8double_to_vector8char_var,
                                          0, 1, 2, 3)) ==
              (LITTLE_END ? 0x03020100 : 0x00010203));
static_assert(__builtin_bit_cast(unsigned long long,
                                 __builtin_shufflevector(
                                     from_vector8double_to_vector8short_var,
                                     from_vector8double_to_vector8short_var, 0,
                                     1, 2, 3)) ==
              (LITTLE_END ? 0x0003000200010000 : 0x0000000100020003));
constexpr vector8double from_vector8float_to_vector8double_var =
    __builtin_convertvector((vector8float){0, 1, 2, 3, 4, 5, 6, 7},
                            vector8double);
constexpr vector8float from_vector8float_to_vector8float_var =
    __builtin_convertvector((vector8float){0, 1, 2, 3, 4, 5, 6, 7},
                            vector8float);
constexpr vector8long from_vector8float_to_vector8long_var =
    __builtin_convertvector((vector8float){0, 1, 2, 3, 4, 5, 6, 7},
                            vector8long);
constexpr vector8int from_vector8float_to_vector8int_var =
    __builtin_convertvector((vector8float){0, 1, 2, 3, 4, 5, 6, 7}, vector8int);
constexpr vector8short from_vector8float_to_vector8short_var =
    __builtin_convertvector((vector8float){0, 1, 2, 3, 4, 5, 6, 7},
                            vector8short);
constexpr vector8char from_vector8float_to_vector8char_var =
    __builtin_convertvector((vector8float){0, 1, 2, 3, 4, 5, 6, 7},
                            vector8char);
constexpr vector8BitInt8 from_vector8float_to_vector8BitInt8_var =
    __builtin_convertvector((vector8float){0, 1, 2, 3, 4, 5, 6, 7},
                            vector8BitInt8);
constexpr vector8BitInt32 from_vector8float_to_vector8BitInt32_var =
    __builtin_convertvector((vector8float){0, 1, 2, 3, 4, 5, 6, 7},
                            vector8BitInt32);
constexpr vector8BitInt128 from_vector8float_to_vector8BitInt128_var =
    __builtin_convertvector((vector8float){0, 1, 2, 3, 4, 5, 6, 7},
                            vector8BitInt128);
static_assert(__builtin_bit_cast(unsigned,
                                 __builtin_shufflevector(
                                     from_vector8float_to_vector8char_var,
                                     from_vector8float_to_vector8char_var, 0, 1,
                                     2, 3)) == (LITTLE_END ? 0x03020100 : 0x00010203));
static_assert(__builtin_bit_cast(
                  unsigned long long,
                  __builtin_shufflevector(from_vector8float_to_vector8short_var,
                                          from_vector8float_to_vector8short_var,
                                          0, 1, 2, 3)) ==
              (LITTLE_END ? 0x0003000200010000 : 0x0000000100020003));
constexpr vector8double from_vector8long_to_vector8double_var =
    __builtin_convertvector((vector8long){0, 1, 2, 3, 4, 5, 6, 7},
                            vector8double);
constexpr vector8float from_vector8long_to_vector8float_var =
    __builtin_convertvector((vector8long){0, 1, 2, 3, 4, 5, 6, 7},
                            vector8float);
constexpr vector8long from_vector8long_to_vector8long_var =
    __builtin_convertvector((vector8long){0, 1, 2, 3, 4, 5, 6, 7}, vector8long);
constexpr vector8int from_vector8long_to_vector8int_var =
    __builtin_convertvector((vector8long){0, 1, 2, 3, 4, 5, 6, 7}, vector8int);
constexpr vector8short from_vector8long_to_vector8short_var =
    __builtin_convertvector((vector8long){0, 1, 2, 3, 4, 5, 6, 7},
                            vector8short);
constexpr vector8char from_vector8long_to_vector8char_var =
    __builtin_convertvector((vector8long){0, 1, 2, 3, 4, 5, 6, 7}, vector8char);
constexpr vector8BitInt8 from_vector8long_to_vector8BitInt8_var =
    __builtin_convertvector((vector8long){0, 1, 2, 3, 4, 5, 6, 7},
                            vector8BitInt8);
constexpr vector8BitInt32 from_vector8long_to_vector8BitInt32_var =
    __builtin_convertvector((vector8long){0, 1, 2, 3, 4, 5, 6, 7},
                            vector8BitInt32);
constexpr vector8BitInt128 from_vector8long_to_vector8BitInt128_var =
    __builtin_convertvector((vector8long){0, 1, 2, 3, 4, 5, 6, 7},
                            vector8BitInt128);
static_assert(__builtin_bit_cast(unsigned,
                                 __builtin_shufflevector(
                                     from_vector8long_to_vector8char_var,
                                     from_vector8long_to_vector8char_var, 0, 1,
                                     2, 3)) == (LITTLE_END ? 0x03020100 : 0x00010203));
static_assert(__builtin_bit_cast(
                  unsigned long long,
                  __builtin_shufflevector(from_vector8long_to_vector8short_var,
                                          from_vector8long_to_vector8short_var,
                                          0, 1, 2, 3)) ==
              (LITTLE_END ? 0x0003000200010000 : 0x0000000100020003));
constexpr vector8double from_vector8int_to_vector8double_var =
    __builtin_convertvector((vector8int){0, 1, 2, 3, 4, 5, 6, 7},
                            vector8double);
constexpr vector8float from_vector8int_to_vector8float_var =
    __builtin_convertvector((vector8int){0, 1, 2, 3, 4, 5, 6, 7}, vector8float);
constexpr vector8long from_vector8int_to_vector8long_var =
    __builtin_convertvector((vector8int){0, 1, 2, 3, 4, 5, 6, 7}, vector8long);
constexpr vector8int from_vector8int_to_vector8int_var =
    __builtin_convertvector((vector8int){0, 1, 2, 3, 4, 5, 6, 7}, vector8int);
constexpr vector8short from_vector8int_to_vector8short_var =
    __builtin_convertvector((vector8int){0, 1, 2, 3, 4, 5, 6, 7}, vector8short);
constexpr vector8char from_vector8int_to_vector8char_var =
    __builtin_convertvector((vector8int){0, 1, 2, 3, 4, 5, 6, 7}, vector8char);
constexpr vector8BitInt8 from_vector8int_to_vector8BitInt8_var =
    __builtin_convertvector((vector8int){0, 1, 2, 3, 4, 5, 6, 7},
                            vector8BitInt8);
constexpr vector8BitInt32 from_vector8int_to_vector8BitInt32_var =
    __builtin_convertvector((vector8int){0, 1, 2, 3, 4, 5, 6, 7},
                            vector8BitInt32);
constexpr vector8BitInt128 from_vector8int_to_vector8BitInt128_var =
    __builtin_convertvector((vector8int){0, 1, 2, 3, 4, 5, 6, 7},
                            vector8BitInt128);
static_assert(__builtin_bit_cast(unsigned,
                                 __builtin_shufflevector(
                                     from_vector8int_to_vector8char_var,
                                     from_vector8int_to_vector8char_var, 0, 1,
                                     2, 3)) == (LITTLE_END ? 0x03020100 : 0x00010203));
static_assert(__builtin_bit_cast(
                  unsigned long long,
                  __builtin_shufflevector(from_vector8int_to_vector8short_var,
                                          from_vector8int_to_vector8short_var,
                                          0, 1, 2, 3)) ==
              (LITTLE_END ? 0x0003000200010000 : 0x0000000100020003));
constexpr vector8double from_vector8short_to_vector8double_var =
    __builtin_convertvector((vector8short){0, 1, 2, 3, 4, 5, 6, 7},
                            vector8double);
constexpr vector8float from_vector8short_to_vector8float_var =
    __builtin_convertvector((vector8short){0, 1, 2, 3, 4, 5, 6, 7},
                            vector8float);
constexpr vector8long from_vector8short_to_vector8long_var =
    __builtin_convertvector((vector8short){0, 1, 2, 3, 4, 5, 6, 7},
                            vector8long);
constexpr vector8int from_vector8short_to_vector8int_var =
    __builtin_convertvector((vector8short){0, 1, 2, 3, 4, 5, 6, 7}, vector8int);
constexpr vector8short from_vector8short_to_vector8short_var =
    __builtin_convertvector((vector8short){0, 1, 2, 3, 4, 5, 6, 7},
                            vector8short);
constexpr vector8char from_vector8short_to_vector8char_var =
    __builtin_convertvector((vector8short){0, 1, 2, 3, 4, 5, 6, 7},
                            vector8char);
constexpr vector8BitInt8 from_vector8short_to_vector8BitInt8_var =
    __builtin_convertvector((vector8short){0, 1, 2, 3, 4, 5, 6, 7},
                            vector8BitInt8);
constexpr vector8BitInt32 from_vector8short_to_vector8BitInt32_var =
    __builtin_convertvector((vector8short){0, 1, 2, 3, 4, 5, 6, 7},
                            vector8BitInt32);
constexpr vector8BitInt128 from_vector8short_to_vector8BitInt128_var =
    __builtin_convertvector((vector8short){0, 1, 2, 3, 4, 5, 6, 7},
                            vector8BitInt128);
static_assert(__builtin_bit_cast(unsigned,
                                 __builtin_shufflevector(
                                     from_vector8short_to_vector8char_var,
                                     from_vector8short_to_vector8char_var, 0, 1,
                                     2, 3)) == (LITTLE_END ? 0x03020100 : 0x00010203));
static_assert(__builtin_bit_cast(
                  unsigned long long,
                  __builtin_shufflevector(from_vector8short_to_vector8short_var,
                                          from_vector8short_to_vector8short_var,
                                          0, 1, 2, 3)) ==
              (LITTLE_END ? 0x0003000200010000 : 0x0000000100020003));
constexpr vector8double from_vector8char_to_vector8double_var =
    __builtin_convertvector((vector8char){0, 1, 2, 3, 4, 5, 6, 7},
                            vector8double);
constexpr vector8float from_vector8char_to_vector8float_var =
    __builtin_convertvector((vector8char){0, 1, 2, 3, 4, 5, 6, 7},
                            vector8float);
constexpr vector8long from_vector8char_to_vector8long_var =
    __builtin_convertvector((vector8char){0, 1, 2, 3, 4, 5, 6, 7}, vector8long);
constexpr vector8int from_vector8char_to_vector8int_var =
    __builtin_convertvector((vector8char){0, 1, 2, 3, 4, 5, 6, 7}, vector8int);
constexpr vector8short from_vector8char_to_vector8short_var =
    __builtin_convertvector((vector8char){0, 1, 2, 3, 4, 5, 6, 7},
                            vector8short);
constexpr vector8char from_vector8char_to_vector8char_var =
    __builtin_convertvector((vector8char){0, 1, 2, 3, 4, 5, 6, 7}, vector8char);
constexpr vector8BitInt8 from_vector8char_to_vector8BitInt8_var =
    __builtin_convertvector((vector8char){0, 1, 2, 3, 4, 5, 6, 7},
                            vector8BitInt8);
constexpr vector8BitInt32 from_vector8char_to_vector8BitInt32_var =
    __builtin_convertvector((vector8char){0, 1, 2, 3, 4, 5, 6, 7},
                            vector8BitInt32);
constexpr vector8BitInt128 from_vector8char_to_vector8BitInt128_var =
    __builtin_convertvector((vector8char){0, 1, 2, 3, 4, 5, 6, 7},
                            vector8BitInt128);
static_assert(__builtin_bit_cast(unsigned,
                                 __builtin_shufflevector(
                                     from_vector8char_to_vector8char_var,
                                     from_vector8char_to_vector8char_var, 0, 1,
                                     2, 3)) == (LITTLE_END ? 0x03020100 : 0x00010203));
static_assert(__builtin_bit_cast(
                  unsigned long long,
                  __builtin_shufflevector(from_vector8char_to_vector8short_var,
                                          from_vector8char_to_vector8short_var,
                                          0, 1, 2, 3)) ==
              (LITTLE_END ? 0x0003000200010000 : 0x0000000100020003));
constexpr vector8double from_vector8BitInt8_to_vector8double_var =
    __builtin_convertvector((vector8BitInt8){0, 1, 2, 3, 4, 5, 6, 7},
                            vector8double);
constexpr vector8float from_vector8BitInt8_to_vector8float_var =
    __builtin_convertvector((vector8BitInt8){0, 1, 2, 3, 4, 5, 6, 7},
                            vector8float);
constexpr vector8long from_vector8BitInt8_to_vector8long_var =
    __builtin_convertvector((vector8BitInt8){0, 1, 2, 3, 4, 5, 6, 7},
                            vector8long);
constexpr vector8int from_vector8BitInt8_to_vector8int_var =
    __builtin_convertvector((vector8BitInt8){0, 1, 2, 3, 4, 5, 6, 7},
                            vector8int);
constexpr vector8short from_vector8BitInt8_to_vector8short_var =
    __builtin_convertvector((vector8BitInt8){0, 1, 2, 3, 4, 5, 6, 7},
                            vector8short);
constexpr vector8char from_vector8BitInt8_to_vector8char_var =
    __builtin_convertvector((vector8BitInt8){0, 1, 2, 3, 4, 5, 6, 7},
                            vector8char);
constexpr vector8BitInt8 from_vector8BitInt8_to_vector8BitInt8_var =
    __builtin_convertvector((vector8BitInt8){0, 1, 2, 3, 4, 5, 6, 7},
                            vector8BitInt8);
constexpr vector8BitInt32 from_vector8BitInt8_to_vector8BitInt32_var =
    __builtin_convertvector((vector8BitInt8){0, 1, 2, 3, 4, 5, 6, 7},
                            vector8BitInt32);
constexpr vector8BitInt128 from_vector8BitInt8_to_vector8BitInt128_var =
    __builtin_convertvector((vector8BitInt8){0, 1, 2, 3, 4, 5, 6, 7},
                            vector8BitInt128);
static_assert(__builtin_bit_cast(unsigned,
                                 __builtin_shufflevector(
                                     from_vector8BitInt8_to_vector8char_var,
                                     from_vector8BitInt8_to_vector8char_var, 0,
                                     1, 2, 3)) ==
              (LITTLE_END ? 0x03020100 : 0x00010203));
static_assert(__builtin_bit_cast(unsigned long long,
                                 __builtin_shufflevector(
                                     from_vector8BitInt8_to_vector8short_var,
                                     from_vector8BitInt8_to_vector8short_var, 0,
                                     1, 2, 3)) ==
              (LITTLE_END ? 0x0003000200010000 : 0x0000000100020003));
constexpr vector8double from_vector8BitInt32_to_vector8double_var =
    __builtin_convertvector((vector8BitInt32){0, 1, 2, 3, 4, 5, 6, 7},
                            vector8double);
constexpr vector8float from_vector8BitInt32_to_vector8float_var =
    __builtin_convertvector((vector8BitInt32){0, 1, 2, 3, 4, 5, 6, 7},
                            vector8float);
constexpr vector8long from_vector8BitInt32_to_vector8long_var =
    __builtin_convertvector((vector8BitInt32){0, 1, 2, 3, 4, 5, 6, 7},
                            vector8long);
constexpr vector8int from_vector8BitInt32_to_vector8int_var =
    __builtin_convertvector((vector8BitInt32){0, 1, 2, 3, 4, 5, 6, 7},
                            vector8int);
constexpr vector8short from_vector8BitInt32_to_vector8short_var =
    __builtin_convertvector((vector8BitInt32){0, 1, 2, 3, 4, 5, 6, 7},
                            vector8short);
constexpr vector8char from_vector8BitInt32_to_vector8char_var =
    __builtin_convertvector((vector8BitInt32){0, 1, 2, 3, 4, 5, 6, 7},
                            vector8char);
constexpr vector8BitInt8 from_vector8BitInt32_to_vector8BitInt8_var =
    __builtin_convertvector((vector8BitInt32){0, 1, 2, 3, 4, 5, 6, 7},
                            vector8BitInt8);
constexpr vector8BitInt32 from_vector8BitInt32_to_vector8BitInt32_var =
    __builtin_convertvector((vector8BitInt32){0, 1, 2, 3, 4, 5, 6, 7},
                            vector8BitInt32);
constexpr vector8BitInt128 from_vector8BitInt32_to_vector8BitInt128_var =
    __builtin_convertvector((vector8BitInt32){0, 1, 2, 3, 4, 5, 6, 7},
                            vector8BitInt128);
static_assert(__builtin_bit_cast(unsigned,
                                 __builtin_shufflevector(
                                     from_vector8BitInt32_to_vector8char_var,
                                     from_vector8BitInt32_to_vector8char_var, 0,
                                     1, 2, 3)) ==
              (LITTLE_END ? 0x03020100 : 0x00010203));
static_assert(__builtin_bit_cast(unsigned long long,
                                 __builtin_shufflevector(
                                     from_vector8BitInt32_to_vector8short_var,
                                     from_vector8BitInt32_to_vector8short_var,
                                     0, 1, 2, 3)) ==
              (LITTLE_END ? 0x0003000200010000 : 0x0000000100020003));
constexpr vector8double from_vector8BitInt128_to_vector8double_var =
    __builtin_convertvector((vector8BitInt128){0, 1, 2, 3, 4, 5, 6, 7},
                            vector8double);
constexpr vector8float from_vector8BitInt128_to_vector8float_var =
    __builtin_convertvector((vector8BitInt128){0, 1, 2, 3, 4, 5, 6, 7},
                            vector8float);
constexpr vector8long from_vector8BitInt128_to_vector8long_var =
    __builtin_convertvector((vector8BitInt128){0, 1, 2, 3, 4, 5, 6, 7},
                            vector8long);
constexpr vector8int from_vector8BitInt128_to_vector8int_var =
    __builtin_convertvector((vector8BitInt128){0, 1, 2, 3, 4, 5, 6, 7},
                            vector8int);
constexpr vector8short from_vector8BitInt128_to_vector8short_var =
    __builtin_convertvector((vector8BitInt128){0, 1, 2, 3, 4, 5, 6, 7},
                            vector8short);
constexpr vector8char from_vector8BitInt128_to_vector8char_var =
    __builtin_convertvector((vector8BitInt128){0, 1, 2, 3, 4, 5, 6, 7},
                            vector8char);
constexpr vector8BitInt8 from_vector8BitInt128_to_vector8BitInt8_var =
    __builtin_convertvector((vector8BitInt128){0, 1, 2, 3, 4, 5, 6, 7},
                            vector8BitInt8);
constexpr vector8BitInt32 from_vector8BitInt128_to_vector8BitInt32_var =
    __builtin_convertvector((vector8BitInt128){0, 1, 2, 3, 4, 5, 6, 7},
                            vector8BitInt32);
constexpr vector8BitInt128 from_vector8BitInt128_to_vector8BitInt128_var =
    __builtin_convertvector((vector8BitInt128){0, 1, 2, 3, 4, 5, 6, 7},
                            vector8BitInt128);
static_assert(__builtin_bit_cast(unsigned,
                                 __builtin_shufflevector(
                                     from_vector8BitInt128_to_vector8char_var,
                                     from_vector8BitInt128_to_vector8char_var,
                                     0, 1, 2, 3)) ==
              (LITTLE_END ? 0x03020100 : 0x00010203));
static_assert(__builtin_bit_cast(unsigned long long,
                                 __builtin_shufflevector(
                                     from_vector8BitInt128_to_vector8short_var,
                                     from_vector8BitInt128_to_vector8short_var,
                                     0, 1, 2, 3)) ==
              (LITTLE_END ? 0x0003000200010000 : 0x0000000100020003));
;
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
constexpr vector4char vectorShuffle3 =
    __builtin_shufflevector(vector4charConst1, vector4charConst2, 0, 2, 4, 6);
static_assert(__builtin_bit_cast(unsigned, vectorShuffle3) ==
              (LITTLE_END ? 0x06040200 : 0x00020406));
constexpr vector8char vectorShuffle4 = __builtin_shufflevector(
    vector8intConst, vector8intConst, 0, 2, 4, 6, 8, 10, 12, 14);
static_assert(__builtin_bit_cast(unsigned long long, vectorShuffle4) ==
              (LITTLE_END ? 0x0E0C0A080E0C0A08 : 0x080A0C0E080A0C0E));
constexpr vector4char vectorShuffle5 =
    __builtin_shufflevector(vector8intConst, vector8intConst, 0, 2, 4, 6);
static_assert(__builtin_bit_cast(unsigned, vectorShuffle5) ==
              (LITTLE_END ? 0x0E0C0A08 : 0x080A0C0E));
constexpr vector8char vectorShuffle6 = __builtin_shufflevector(
    vector4charConst1, vector4charConst2, 0, 2, 4, 6, 1, 3, 5, 7);
static_assert(__builtin_bit_cast(unsigned long long, vectorShuffle6) ==
              (LITTLE_END ? 0x0705030106040200 : 0x0002040601030507));

constexpr vector4char
    vectorShuffleFail1 = // expected-error {{constexpr variable 'vectorShuffleFail1'\
 must be initialized by a constant expression}}
    __builtin_shufflevector( // expected-error {{index for __builtin_shufflevector \
not within the bounds of the input vectors; index of -1 found at position 0 is not \
permitted in a constexpr context}}
        vector4charConst1,
        vector4charConst2, -1, -1, -1, -1);

static_assert(__builtin_reduce_add((vector4char){}) == 0);
static_assert(__builtin_reduce_add((vector4char){1, 2, 3, 4}) == 10);
static_assert(__builtin_reduce_add((vector4short){10, 20, 30, 40}) == 100);
static_assert(__builtin_reduce_add((vector4int){100, 200, 300, 400}) == 1000);
static_assert(__builtin_reduce_add((vector4long){1000, 2000, 3000, 4000}) == 10000);
constexpr int reduceAddInt1 = __builtin_reduce_add((vector4int){~(1 << 31), 0, 0, 1});
// expected-error@-1 {{must be initialized by a constant expression}} \
// expected-note@-1 {{outside the range of representable values of type 'int'}}
constexpr long long reduceAddLong1 = __builtin_reduce_add((vector4long){~(1LL << 63), 0, 0, 1});
// expected-error@-1 {{must be initialized by a constant expression}} \
// expected-note@-1 {{outside the range of representable values of type 'long long'}}
constexpr int reduceAddInt2 = __builtin_reduce_add((vector4int){(1 << 31), 0, 0, -1});
// expected-error@-1 {{must be initialized by a constant expression}} \
// expected-note@-1 {{outside the range of representable values of type 'int'}}
constexpr long long reduceAddLong2 = __builtin_reduce_add((vector4long){(1LL << 63), 0, 0, -1});
// expected-error@-1 {{must be initialized by a constant expression}} \
// expected-note@-1 {{outside the range of representable values of type 'long long'}}
static_assert(__builtin_reduce_add((vector4uint){~0U, 0, 0, 1}) == 0);
static_assert(__builtin_reduce_add((vector4ulong){~0ULL, 0, 0, 1}) == 0);

static_assert(__builtin_reduce_mul((vector4char){}) == 0);
static_assert(__builtin_reduce_mul((vector4char){1, 2, 3, 4}) == 24);
static_assert(__builtin_reduce_mul((vector4short){1, 2, 30, 40}) == 2400);
static_assert(__builtin_reduce_mul((vector4int){10, 20, 300, 400}) == 24000000);
static_assert(__builtin_reduce_mul((vector4long){1000L, 2000L, 3000L, 4000L}) == 24000000000000L);
constexpr int reduceMulInt1 = __builtin_reduce_mul((vector4int){~(1 << 31), 1, 1, 2});
// expected-error@-1 {{must be initialized by a constant expression}} \
// expected-note@-1 {{outside the range of representable values of type 'int'}}
constexpr long long reduceMulLong1 = __builtin_reduce_mul((vector4long){~(1LL << 63), 1, 1, 2});
// expected-error@-1 {{must be initialized by a constant expression}} \
// expected-note@-1 {{outside the range of representable values of type 'long long'}}
constexpr int reduceMulInt2 = __builtin_reduce_mul((vector4int){(1 << 31), 1, 1, 2});
// expected-error@-1 {{must be initialized by a constant expression}} \
// expected-note@-1 {{outside the range of representable values of type 'int'}}
constexpr long long reduceMulLong2 = __builtin_reduce_mul((vector4long){(1LL << 63), 1, 1, 2});
// expected-error@-1 {{must be initialized by a constant expression}} \
// expected-note@-1 {{outside the range of representable values of type 'long long'}}
static_assert(__builtin_reduce_mul((vector4uint){~0U, 1, 1, 2}) == ~0U - 1);
static_assert(__builtin_reduce_mul((vector4ulong){~0ULL, 1, 1, 2}) == ~0ULL - 1);

static_assert(__builtin_reduce_and((vector4char){}) == 0);
static_assert(__builtin_reduce_and((vector4char){(char)0x11, (char)0x22, (char)0x44, (char)0x88}) == 0);
static_assert(__builtin_reduce_and((vector4short){(short)0x1111, (short)0x2222, (short)0x4444, (short)0x8888}) == 0);
static_assert(__builtin_reduce_and((vector4int){(int)0x11111111, (int)0x22222222, (int)0x44444444, (int)0x88888888}) == 0);
static_assert(__builtin_reduce_and((vector4long){(long long)0x1111111111111111L, (long long)0x2222222222222222L, (long long)0x4444444444444444L, (long long)0x8888888888888888L}) == 0L);
static_assert(__builtin_reduce_and((vector4char){(char)-1, (char)~0x22, (char)~0x44, (char)~0x88}) == 0x11);
static_assert(__builtin_reduce_and((vector4short){(short)~0x1111, (short)-1, (short)~0x4444, (short)~0x8888}) == 0x2222);
static_assert(__builtin_reduce_and((vector4int){(int)~0x11111111, (int)~0x22222222, (int)-1, (int)~0x88888888}) == 0x44444444);
static_assert(__builtin_reduce_and((vector4long){(long long)~0x1111111111111111L, (long long)~0x2222222222222222L, (long long)~0x4444444444444444L, (long long)-1}) == 0x8888888888888888L);
static_assert(__builtin_reduce_and((vector4uint){0x11111111U, 0x22222222U, 0x44444444U, 0x88888888U}) == 0U);
static_assert(__builtin_reduce_and((vector4ulong){0x1111111111111111UL, 0x2222222222222222UL, 0x4444444444444444UL, 0x8888888888888888UL}) == 0L);

static_assert(__builtin_reduce_or((vector4char){}) == 0);
static_assert(__builtin_reduce_or((vector4char){(char)0x11, (char)0x22, (char)0x44, (char)0x88}) == (char)0xFF);
static_assert(__builtin_reduce_or((vector4short){(short)0x1111, (short)0x2222, (short)0x4444, (short)0x8888}) == (short)0xFFFF);
static_assert(__builtin_reduce_or((vector4int){(int)0x11111111, (int)0x22222222, (int)0x44444444, (int)0x88888888}) == (int)0xFFFFFFFF);
static_assert(__builtin_reduce_or((vector4long){(long long)0x1111111111111111L, (long long)0x2222222222222222L, (long long)0x4444444444444444L, (long long)0x8888888888888888L}) == (long long)0xFFFFFFFFFFFFFFFFL);
static_assert(__builtin_reduce_or((vector4char){(char)0, (char)0x22, (char)0x44, (char)0x88}) == ~0x11);
static_assert(__builtin_reduce_or((vector4short){(short)0x1111, (short)0, (short)0x4444, (short)0x8888}) == ~0x2222);
static_assert(__builtin_reduce_or((vector4int){(int)0x11111111, (int)0x22222222, (int)0, (int)0x88888888}) == ~0x44444444);
static_assert(__builtin_reduce_or((vector4long){(long long)0x1111111111111111L, (long long)0x2222222222222222L, (long long)0x4444444444444444L, (long long)0}) == ~0x8888888888888888L);
static_assert(__builtin_reduce_or((vector4uint){0x11111111U, 0x22222222U, 0x44444444U, 0x88888888U}) == 0xFFFFFFFFU);
static_assert(__builtin_reduce_or((vector4ulong){0x1111111111111111UL, 0x2222222222222222UL, 0x4444444444444444UL, 0x8888888888888888UL}) == 0xFFFFFFFFFFFFFFFFL);

static_assert(__builtin_reduce_xor((vector4char){}) == 0);
static_assert(__builtin_reduce_xor((vector4char){(char)0x11, (char)0x22, (char)0x44, (char)0x88}) == (char)0xFF);
static_assert(__builtin_reduce_xor((vector4short){(short)0x1111, (short)0x2222, (short)0x4444, (short)0x8888}) == (short)0xFFFF);
static_assert(__builtin_reduce_xor((vector4int){(int)0x11111111, (int)0x22222222, (int)0x44444444, (int)0x88888888}) == (int)0xFFFFFFFF);
static_assert(__builtin_reduce_xor((vector4long){(long long)0x1111111111111111L, (long long)0x2222222222222222L, (long long)0x4444444444444444L, (long long)0x8888888888888888L}) == (long long)0xFFFFFFFFFFFFFFFFL);
static_assert(__builtin_reduce_xor((vector4uint){0x11111111U, 0x22222222U, 0x44444444U, 0x88888888U}) == 0xFFFFFFFFU);
static_assert(__builtin_reduce_xor((vector4ulong){0x1111111111111111UL, 0x2222222222222222UL, 0x4444444444444444UL, 0x8888888888888888UL}) == 0xFFFFFFFFFFFFFFFFUL);

static_assert(__builtin_reduce_min((vector4char){}) == 0);
static_assert(__builtin_reduce_min((vector4char){(char)0x11, (char)0x22, (char)0x44, (char)0x88}) == (char)0x88);
static_assert(__builtin_reduce_min((vector4short){(short)0x1111, (short)0x2222, (short)0x4444, (short)0x8888}) == (short)0x8888);
static_assert(__builtin_reduce_min((vector4int){(int)0x11111111, (int)0x22222222, (int)0x44444444, (int)0x88888888}) == (int)0x88888888);
static_assert(__builtin_reduce_min((vector4long){(long long)0x1111111111111111L, (long long)0x2222222222222222L, (long long)0x4444444444444444L, (long long)0x8888888888888888L}) == (long long)0x8888888888888888L);
static_assert(__builtin_reduce_min((vector4uint){0x11111111U, 0x22222222U, 0x44444444U, 0x88888888U}) == 0x11111111U);
static_assert(__builtin_reduce_min((vector4ulong){0x1111111111111111UL, 0x2222222222222222UL, 0x4444444444444444UL, 0x8888888888888888UL}) == 0x1111111111111111UL);
static_assert(__builtin_reduce_max((vector4char){}) == 0);
static_assert(__builtin_reduce_max((vector4char){(char)0x11, (char)0x22, (char)0x44, (char)0x88}) == (char)0x44);
static_assert(__builtin_reduce_max((vector4short){(short)0x1111, (short)0x2222, (short)0x4444, (short)0x8888}) == (short)0x4444);
static_assert(__builtin_reduce_max((vector4int){(int)0x11111111, (int)0x22222222, (int)0x44444444, (int)0x88888888}) == (int)0x44444444);
static_assert(__builtin_reduce_max((vector4long){(long long)0x1111111111111111L, (long long)0x2222222222222222L, (long long)0x4444444444444444L, (long long)0x8888888888888888L}) == (long long)0x4444444444444444L);
static_assert(__builtin_reduce_max((vector4uint){0x11111111U, 0x22222222U, 0x44444444U, 0x88888888U}) == 0x88888888U);
static_assert(__builtin_reduce_max((vector4ulong){0x1111111111111111UL, 0x2222222222222222UL, 0x4444444444444444UL, 0x8888888888888888UL}) == 0x8888888888888888UL);

static_assert(__builtin_bit_cast(unsigned, __builtin_elementwise_popcount((vector4char){1, 2, 3, 4})) == (LITTLE_END ? 0x01020101 : 0x01010201));
static_assert(__builtin_bit_cast(unsigned long long, __builtin_elementwise_popcount((vector4short){0, 0x0F0F, ~0, ~0x0F0F})) == (LITTLE_END ? 0x0008001000080000 : 0x0000000800100008));
static_assert(__builtin_reduce_add(__builtin_elementwise_popcount((vector4int){1, 2, 3, 4})) == 5);
static_assert(__builtin_reduce_add(__builtin_elementwise_popcount((vector4int){0, 0xF0F0, ~0, ~0xF0F0})) == 16 * sizeof(int));
static_assert(__builtin_reduce_add(__builtin_elementwise_popcount((vector4long){1L, 2L, 3L, 4L})) == 5L);
static_assert(__builtin_reduce_add(__builtin_elementwise_popcount((vector4long){0L, 0xF0F0L, ~0L, ~0xF0F0L})) == 16 * sizeof(long long));
static_assert(__builtin_reduce_add(__builtin_elementwise_popcount((vector4uint){1U, 2U, 3U, 4U})) == 5U);
static_assert(__builtin_reduce_add(__builtin_elementwise_popcount((vector4uint){0U, 0xF0F0U, ~0U, ~0xF0F0U})) == 16 * sizeof(int));
static_assert(__builtin_reduce_add(__builtin_elementwise_popcount((vector4ulong){1UL, 2UL, 3UL, 4UL})) == 5UL);
static_assert(__builtin_reduce_add(__builtin_elementwise_popcount((vector4ulong){0ULL, 0xF0F0ULL, ~0ULL, ~0xF0F0ULL})) == 16 * sizeof(unsigned long long));
static_assert(__builtin_elementwise_popcount(0) == 0);
static_assert(__builtin_elementwise_popcount(0xF0F0) == 8);
static_assert(__builtin_elementwise_popcount(~0) == 8 * sizeof(int));
static_assert(__builtin_elementwise_popcount(0U) == 0);
static_assert(__builtin_elementwise_popcount(0xF0F0U) == 8);
static_assert(__builtin_elementwise_popcount(~0U) == 8 * sizeof(int));
static_assert(__builtin_elementwise_popcount(0L) == 0);
static_assert(__builtin_elementwise_popcount(0xF0F0L) == 8);
static_assert(__builtin_elementwise_popcount(~0LL) == 8 * sizeof(long long));

static_assert(__builtin_elementwise_bitreverse(0x12345678) == 0x1E6A2C48);
static_assert(__builtin_elementwise_bitreverse(0x0123456789ABCDEFULL) == 0xF7B3D591E6A2C480);
static_assert(__builtin_bit_cast(unsigned, __builtin_elementwise_bitreverse((vector4char){1, 2, 4, 8})) == (LITTLE_END ? 0x10204080 : 0x80402010));
static_assert(__builtin_bit_cast(unsigned long long, __builtin_elementwise_bitreverse((vector4short){1, 2, 4, 8})) == (LITTLE_END ? 0x1000200040008000 : 0x8000400020001000));

static_assert(__builtin_elementwise_add_sat(1, 2) == 3);
static_assert(__builtin_elementwise_add_sat(1U, 2U) == 3U);
static_assert(__builtin_elementwise_add_sat(~(1 << 31), 42) == ~(1 << 31));
static_assert(__builtin_elementwise_add_sat((1 << 31), -42) == (1 << 31));
static_assert(__builtin_elementwise_add_sat(~0U, 1U) == ~0U);
static_assert(__builtin_bit_cast(unsigned, __builtin_elementwise_add_sat((vector4char){1, 2, 3, 4}, (vector4char){1, 2, 3, 4})) == (LITTLE_END ? 0x08060402 : 0x02040608));
static_assert(__builtin_bit_cast(unsigned long long, __builtin_elementwise_add_sat((vector4short){(short)0x8000, (short)0x8001, (short)0x8002, (short)0x8003}, (vector4short){-7, -8, -9, -10}) == (LITTLE_END ? 0x8000800080008000 : 0x8000800080008000)));

static_assert(__builtin_elementwise_sub_sat(1, 2) == -1);
static_assert(__builtin_elementwise_sub_sat(2U, 1U) == 1U);
static_assert(__builtin_elementwise_sub_sat(~(1 << 31), -42) == ~(1 << 31));
static_assert(__builtin_elementwise_sub_sat((1 << 31), 42) == (1 << 31));
static_assert(__builtin_elementwise_sub_sat(0U, 1U) == 0U);
static_assert(__builtin_bit_cast(unsigned, __builtin_elementwise_sub_sat((vector4char){5, 4, 3, 2}, (vector4char){1, 1, 1, 1})) == (LITTLE_END ? 0x01020304 : 0x04030201));
static_assert(__builtin_bit_cast(unsigned long long, __builtin_elementwise_sub_sat((vector4short){(short)0x8000, (short)0x8001, (short)0x8002, (short)0x8003}, (vector4short){7, 8, 9, 10}) == (LITTLE_END ? 0x8000800080008000 : 0x8000800080008000)));
