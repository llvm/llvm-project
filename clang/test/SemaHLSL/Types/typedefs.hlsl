// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.4-library -finclude-default-header -verify -fnative-half-type %s
// RUN: %clang_cc1 -triple spirv-linux-vulkan-library -finclude-default-header -verify -fnative-half-type %s

// expected-no-diagnostics
#define SizeCheck(Ty, SizeInBits)                                              \
  _Static_assert(sizeof(Ty) == SizeInBits / 8, #Ty " is " #SizeInBits "-bit"); \
  _Static_assert(sizeof(Ty##1) == (SizeInBits * 1) / 8, #Ty "1 is 1x" #SizeInBits "-bit"); \
  _Static_assert(__builtin_vectorelements(Ty##1) == 1, #Ty "1 is has 1 " #SizeInBits "-bit element"); \
  _Static_assert(sizeof(Ty##2) == (SizeInBits * 2) / 8, #Ty "2 is 2x" #SizeInBits "-bit"); \
  _Static_assert(__builtin_vectorelements(Ty##2) == 2, #Ty "2 is has 2 " #SizeInBits "-bit element"); \
  _Static_assert(__builtin_vectorelements(Ty##3) == 3, #Ty "3 is has 3 " #SizeInBits "-bit element"); \
  _Static_assert(sizeof(Ty##4) == (SizeInBits * 4) / 8, #Ty "4 is 4x" #SizeInBits "-bit"); \
  _Static_assert(__builtin_vectorelements(Ty##4) == 4, #Ty "4 is has 4 " #SizeInBits "-bit element");

// FIXME: https://github.com/llvm/llvm-project/issues/104503 - 3 element vectors
// should be the size of 3 elements not padded to 4.
// _Static_assert(sizeof(Ty##3) == (SizeInBits * 3) / 8, #Ty "3 is 3x" #SizeInBits "-bit");

SizeCheck(int16_t, 16);
SizeCheck(uint16_t, 16);
SizeCheck(half, 16);
SizeCheck(float16_t, 16);

SizeCheck(int, 32);
SizeCheck(uint, 32);
SizeCheck(int32_t, 32);
SizeCheck(uint32_t, 32);
SizeCheck(float, 32);
SizeCheck(float32_t, 32);

SizeCheck(int64_t, 64);
SizeCheck(uint64_t, 64);
SizeCheck(double, 64);
SizeCheck(float64_t, 64);
