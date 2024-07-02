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
