// RUN: %clang_cc1 -ffreestanding -emit-llvm-only  -triple x86_64-windows-coff -fdump-record-layouts %s | FileCheck %s --check-prefix=CHECK-MS
// RUN: %clang_cc1 -ffreestanding -emit-llvm-only  -triple x86_64-apple-darwin -fdump-record-layouts %s | FileCheck %s --check-prefix=CHECK-NOTMS
#include <x86intrin.h>

#pragma pack(1)

struct s_m64 {
  int a;
  __m64 b;
};
typedef struct s_m64 m64;

#if defined(_WIN32)
static int a1[(sizeof(m64) == 16) - 1];
#else
static int a1[(sizeof(m64) == 12) - 1];
#endif

struct s_m128 {
  int a;
  __m128 b;
};
typedef struct s_m128 m128;

#if defined(_WIN32)
static int a1[(sizeof(m128) == 32) - 1];
#else
static int a1[(sizeof(m128) == 20) - 1];
#endif

struct s_m128i {
  int a;
  __m128i b;
};
typedef struct s_m128i m128i;

#if defined(_WIN32)
static int a1[(sizeof(m128i) == 32) - 1];
#else
static int a1[(sizeof(m128i) == 20) - 1];
#endif

struct s_m128d {
  int a;
  __m128d b;
};
typedef struct s_m128d m128d;

#if defined(_WIN32)
static int a1[(sizeof(m128d) == 32) - 1];
#else
static int a1[(sizeof(m128d) == 20) - 1];
#endif

struct s_m256 {
  int a;
  __m256 b;
};
typedef struct s_m256 m256;

#if defined(_WIN32)
static int a1[(sizeof(m256) == 64) - 1];
#else
static int a1[(sizeof(m256) == 36) - 1];
#endif

struct s_m256i {
  int a;
  __m256i b;
};
typedef struct s_m256i m256i;

#if defined(_WIN32)
static int a1[(sizeof(m256i) == 64) - 1];
#else
static int a1[(sizeof(m256i) == 36) - 1];
#endif

struct s_m256d {
  int a;
  __m256d b;
};
typedef struct s_m256d m256d;

#if defined(_WIN32)
static int a1[(sizeof(m256d) == 64) - 1];
#else
static int a1[(sizeof(m256d) == 36) - 1];
#endif

struct s_m512 {
  int a;
  __m512 b;
};
typedef struct s_m512 m512;

#if defined(_WIN32)
static int a1[(sizeof(m512) == 128) - 1];
#else
static int a1[(sizeof(m512) == 68) - 1];
#endif

struct s_m512i {
  int a;
  __m512i b;
};
typedef struct s_m512i m512i;

#if defined(_WIN32)
static int a1[(sizeof(m512i) == 128) - 1];
#else
static int a1[(sizeof(m512i) == 68) - 1];
#endif

struct s_m512d {
  int a;
  __m512d b;
};
typedef struct s_m512d m512d;

#if defined(_WIN32)
static int a1[(sizeof(m512d) == 128) - 1];
#else
static int a1[(sizeof(m512d) == 68) - 1];
#endif

// CHECK-MS: *** Dumping AST Record Layout
// CHECK-MS:          0 | struct s_m64
// CHECK-MS:          0 |   int a
// CHECK-MS:          8 |   __m64 b
// CHECK-MS:            | [sizeof=16, align=8]
// CHECK-MS: *** Dumping AST Record Layout
// CHECK-MS:          0 | struct s_m128
// CHECK-MS:          0 |   int a
// CHECK-MS:         16 |   __m128 b
// CHECK-MS:            | [sizeof=32, align=16]
// CHECK-MS: *** Dumping AST Record Layout
// CHECK-MS:          0 | struct s_m128i
// CHECK-MS:          0 |   int a
// CHECK-MS:         16 |   __m128i b
// CHECK-MS:            | [sizeof=32, align=16]
// CHECK-MS: *** Dumping AST Record Layout
// CHECK-MS:          0 | struct s_m128d
// CHECK-MS:          0 |   int a
// CHECK-MS:         16 |   __m128d b
// CHECK-MS:            | [sizeof=32, align=16]
// CHECK-MS: *** Dumping AST Record Layout
// CHECK-MS:          0 | struct s_m256
// CHECK-MS:          0 |   int a
// CHECK-MS:         32 |   __m256 b
// CHECK-MS:            | [sizeof=64, align=32]
// CHECK-MS: *** Dumping AST Record Layout
// CHECK-MS:          0 | struct s_m256i
// CHECK-MS:          0 |   int a
// CHECK-MS:         32 |   __m256i b
// CHECK-MS:            | [sizeof=64, align=32]
// CHECK-MS: *** Dumping AST Record Layout
// CHECK-MS:          0 | struct s_m256d
// CHECK-MS:          0 |   int a
// CHECK-MS:         32 |   __m256d b
// CHECK-MS:            | [sizeof=64, align=32]
// CHECK-MS: *** Dumping AST Record Layout
// CHECK-MS:          0 | struct s_m512
// CHECK-MS:          0 |   int a
// CHECK-MS:         64 |   __m512 b
// CHECK-MS:            | [sizeof=128, align=64]
// CHECK-MS: *** Dumping AST Record Layout
// CHECK-MS:          0 | struct s_m512i
// CHECK-MS:          0 |   int a
// CHECK-MS:         64 |   __m512i b
// CHECK-MS:            | [sizeof=128, align=64]
// CHECK-MS: *** Dumping AST Record Layout
// CHECK-MS:          0 | struct s_m512d
// CHECK-MS:          0 |   int a
// CHECK-MS:         64 |   __m512d b
// CHECK-MS:            | [sizeof=128, align=64]

// CHECK-NOTMS: *** Dumping AST Record Layout
// CHECK-NOTMS:          0 | struct s_m64
// CHECK-NOTMS:          0 |   int a
// CHECK-NOTMS:          4 |   __m64 b
// CHECK-NOTMS:            | [sizeof=12, align=1]
// CHECK-NOTMS: *** Dumping AST Record Layout
// CHECK-NOTMS:          0 | struct s_m128
// CHECK-NOTMS:          0 |   int a
// CHECK-NOTMS:          4 |   __m128 b
// CHECK-NOTMS:            | [sizeof=20, align=1]
// CHECK-NOTMS: *** Dumping AST Record Layout
// CHECK-NOTMS:          0 | struct s_m128i
// CHECK-NOTMS:          0 |   int a
// CHECK-NOTMS:          4 |   __m128i b
// CHECK-NOTMS:            | [sizeof=20, align=1]
// CHECK-NOTMS: *** Dumping AST Record Layout
// CHECK-NOTMS:          0 | struct s_m128d
// CHECK-NOTMS:          0 |   int a
// CHECK-NOTMS:          4 |   __m128d b
// CHECK-NOTMS:            | [sizeof=20, align=1]
// CHECK-NOTMS: *** Dumping AST Record Layout
// CHECK-NOTMS:          0 | struct s_m256
// CHECK-NOTMS:          0 |   int a
// CHECK-NOTMS:          4 |   __m256 b
// CHECK-NOTMS:            | [sizeof=36, align=1]
// CHECK-NOTMS: *** Dumping AST Record Layout
// CHECK-NOTMS:          0 | struct s_m256i
// CHECK-NOTMS:          0 |   int a
// CHECK-NOTMS:          4 |   __m256i b
// CHECK-NOTMS:            | [sizeof=36, align=1]
// CHECK-NOTMS: *** Dumping AST Record Layout
// CHECK-NOTMS:          0 | struct s_m256d
// CHECK-NOTMS:          0 |   int a
// CHECK-NOTMS:          4 |   __m256d b
// CHECK-NOTMS:            | [sizeof=36, align=1]
// CHECK-NOTMS: *** Dumping AST Record Layout
// CHECK-NOTMS:          0 | struct s_m512
// CHECK-NOTMS:          0 |   int a
// CHECK-NOTMS:          4 |   __m512 b
// CHECK-NOTMS:            | [sizeof=68, align=1]
// CHECK-NOTMS: *** Dumping AST Record Layout
// CHECK-NOTMS:          0 | struct s_m512i
// CHECK-NOTMS:          0 |   int a
// CHECK-NOTMS:          4 |   __m512i b
// CHECK-NOTMS:            | [sizeof=68, align=1]
// CHECK-NOTMS: *** Dumping AST Record Layout
// CHECK-NOTMS:          0 | struct s_m512d
// CHECK-NOTMS:          0 |   int a
// CHECK-NOTMS:          4 |   __m512d b
// CHECK-NOTMS:            | [sizeof=68, align=1]
