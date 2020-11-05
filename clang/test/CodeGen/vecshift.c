// RUN: %clang_cc1  -Wno-error=vec-elem-size -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1  -Wno-error=vec-elem-size -DEXT -emit-llvm %s -o - | FileCheck %s

#ifdef EXT
typedef __attribute__((__ext_vector_type__(8))) char vector_char8;
typedef __attribute__((__ext_vector_type__(8))) short vector_short8;
typedef __attribute__((__ext_vector_type__(8))) int vector_int8;
typedef __attribute__((__ext_vector_type__(8))) unsigned char vector_uchar8;
typedef __attribute__((__ext_vector_type__(8))) unsigned short vector_ushort8;
typedef __attribute__((__ext_vector_type__(8))) unsigned int vector_uint8;
typedef __attribute__((__ext_vector_type__(4))) char vector_char4;
typedef __attribute__((__ext_vector_type__(4))) short vector_short4;
typedef __attribute__((__ext_vector_type__(4))) int vector_int4;
typedef __attribute__((__ext_vector_type__(4))) unsigned char vector_uchar4;
typedef __attribute__((__ext_vector_type__(4))) unsigned short vector_ushort4;
typedef __attribute__((__ext_vector_type__(4))) unsigned int vector_uint4;
#else
typedef __attribute__((vector_size(8))) char vector_char8;
typedef __attribute__((vector_size(16))) short vector_short8;
typedef __attribute__((vector_size(32))) int vector_int8;
typedef __attribute__((vector_size(8))) unsigned char vector_uchar8;
typedef __attribute__((vector_size(16))) unsigned short vector_ushort8;
typedef __attribute__((vector_size(32))) unsigned int vector_uint8;
typedef __attribute__((vector_size(4))) char vector_char4;
typedef __attribute__((vector_size(4))) short vector_short4;
typedef __attribute__((vector_size(16))) int vector_int4;
typedef __attribute__((vector_size(4))) unsigned char vector_uchar4;
typedef __attribute__((vector_size(8))) unsigned short vector_ushort4;
typedef __attribute__((vector_size(16))) unsigned int vector_uint4;
#endif

char c;
short s;
int i;
unsigned char uc;
unsigned short us;
unsigned int ui;
vector_char8 vc8;
vector_short8 vs8;
vector_int8 vi8;
vector_uchar8 vuc8;
vector_ushort8 vus8;
vector_uint8 vui8;
vector_char4 vc4;
vector_short4 vs4;
vector_int4 vi4;
vector_uchar4 vuc4;
vector_ushort4 vus4;
vector_uint4 vui4;

void foo() {
  vc8 = 1 << vc8;
// CHECK: [[t0:%.+]] = load <8 x i8>, <8 x i8>* {{@.+}},
// CHECK: shl <8 x i8> <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>, [[t0]]
  vuc8 = 1 << vuc8;
// CHECK: [[t1:%.+]] = load <8 x i8>, <8 x i8>* {{@.+}},
// CHECK: shl <8 x i8> <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>, [[t1]]
  vi8 = 1 << vi8;
// CHECK: [[t2:%.+]] = load <8 x i32>, <8 x i32>* {{@.+}},
// CHECK: shl <8 x i32> <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>, [[t2]]
  vui8 = 1 << vui8;
// CHECK: [[t3:%.+]] = load <8 x i32>, <8 x i32>* {{@.+}},
// CHECK: shl <8 x i32> <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>, [[t3]]
  vs8 = 1 << vs8;
// CHECK: [[t4:%.+]] = load <8 x i16>, <8 x i16>* {{@.+}},
// CHECK: shl <8 x i16> <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>, [[t4]]
  vus8 = 1 << vus8;
// CHECK: [[t5:%.+]] = load <8 x i16>, <8 x i16>* {{@.+}},
// CHECK: shl <8 x i16> <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>, [[t5]]

  vc8 = c << vc8;
// CHECK: [[t6:%.+]] = load i8, i8* @c,
// CHECK: [[splat_splatinsert:%.+]] = insertelement <8 x i8> undef, i8 [[t6]], i32 0
// CHECK: [[splat_splat:%.+]] = shufflevector <8 x i8> [[splat_splatinsert]], <8 x i8> undef, <8 x i32> zeroinitializer
// CHECK: [[t7:%.+]] = load <8 x i8>, <8 x i8>* {{@.+}},
// CHECK: shl <8 x i8> [[splat_splat]], [[t7]]
  vuc8 = i << vuc8;
// CHECK: [[t8:%.+]] = load i32, i32* @i,
// CHECK: [[tconv:%.+]] = trunc i32 [[t8]] to i8
// CHECK: [[splat_splatinsert7:%.+]] = insertelement <8 x i8> undef, i8 [[tconv]], i32 0
// CHECK: [[splat_splat8:%.+]] = shufflevector <8 x i8> [[splat_splatinsert7]], <8 x i8> undef, <8 x i32> zeroinitializer
// CHECK: [[t9:%.+]] = load <8 x i8>, <8 x i8>* {{@.+}},
// CHECK: shl <8 x i8> [[splat_splat8]], [[t9]]
  vi8 = uc << vi8;
// CHECK: [[t10:%.+]] = load i8, i8* @uc,
// CHECK: [[conv10:%.+]] = zext i8 [[t10]] to i32
// CHECK: [[splat_splatinsert11:%.+]] = insertelement <8 x i32> undef, i32 [[conv10]], i32 0
// CHECK: [[splat_splat12:%.+]] = shufflevector <8 x i32> [[splat_splatinsert11]], <8 x i32> undef, <8 x i32> zeroinitializer
// CHECK: [[t11:%.+]] = load <8 x i32>, <8 x i32>* {{@.+}},
// CHECK: shl <8 x i32> [[splat_splat12]], [[t11]]
  vui8 = us << vui8;
// CHECK: [[t12:%.+]] = load i16, i16* @us,
// CHECK: [[conv14:%.+]] = zext i16 [[t12]] to i32
// CHECK: [[splat_splatinsert15:%.+]] = insertelement <8 x i32> undef, i32 [[conv14]], i32 0
// CHECK: [[splat_splat16:%.+]] = shufflevector <8 x i32> [[splat_splatinsert15]], <8 x i32> undef, <8 x i32> zeroinitializer
// CHECK: [[t13:%.+]] = load <8 x i32>, <8 x i32>* {{@.+}},
// CHECK: shl <8 x i32> [[splat_splat16]], [[t13]]
  vs8 = ui << vs8;
// CHECK: [[t14:%.+]] = load i32, i32* @ui,
// CHECK: [[conv18:%.+]] = trunc i32 [[t14]] to i16
// CHECK: [[splat_splatinsert19:%.+]] = insertelement <8 x i16> undef, i16 [[conv18]], i32 0
// CHECK: [[splat_splat20:%.+]] = shufflevector <8 x i16> [[splat_splatinsert19]], <8 x i16> undef, <8 x i32> zeroinitializer
// CHECK: [[t15:%.+]] = load <8 x i16>, <8 x i16>* {{@.+}},
// CHECK: shl <8 x i16> [[splat_splat20]], [[t15]]
  vus8 = 1 << vus8;
// CHECK: [[t16:%.+]] = load <8 x i16>, <8 x i16>* {{@.+}},
// CHECK: [[shl22:%.+]] = shl <8 x i16> <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>, [[t16]]

 vc8 = vc8 << vc8;
// CHECK: [[t17:%.+]] = load <8 x i8>, <8 x i8>* {{@.+}},
// CHECK: [[t18:%.+]] = load <8 x i8>, <8 x i8>* {{@.+}},
// CHECK: shl <8 x i8> [[t17]], [[t18]]
  vi8 = vi8 << vuc8;
// CHECK: [[t19:%.+]] = load <8 x i32>, <8 x i32>* {{@.+}},
// CHECK: [[t20:%.+]] = load <8 x i8>, <8 x i8>* {{@.+}},
// CHECK: [[shprom:%.+]] = zext <8 x i8> [[t20]] to <8 x i32>
// CHECK: shl <8 x i32> [[t19]], [[shprom]]
  vuc8 = vuc8 << vi8;
// CHECK: [[t21:%.+]] = load <8 x i8>, <8 x i8>* {{@.+}},
// CHECK: [[t22:%.+]] = load <8 x i32>, <8 x i32>* {{@.+}},
// CHECK: [[sh_prom25:%.+]] = trunc <8 x i32> [[t22]] to <8 x i8>
// CHECK: shl <8 x i8> [[t21]], [[sh_prom25]]
  vus8 = vus8 << vui8;
// CHECK: [[t23:%.+]] = load <8 x i16>, <8 x i16>* {{@.+}},
// CHECK: [[t24:%.+]] = load <8 x i32>, <8 x i32>* {{@.+}},
// CHECK: [[sh_prom27:%.+]] = trunc <8 x i32> [[t24]] to <8 x i16>
// CHECK: shl <8 x i16> [[t23]], [[sh_prom27]]
  vui8 = vui8 << vs8;
// CHECK: [[t25:%.+]] = load <8 x i32>, <8 x i32>* {{@.+}},
// CHECK: [[t26:%.+]] = load <8 x i16>, <8 x i16>* {{@.+}},
// CHECK: [[sh_prom29:%.+]] = zext <8 x i16> [[t26]] to <8 x i32>
// CHECK: shl <8 x i32> [[t25]], [[sh_prom29]]

  vui8 <<= s;
// CHECK: [[t27:%.+]] = load i16, i16* @s,
// CHECK: [[conv40:%.+]] = sext i16 [[t27]] to i32
// CHECK: [[splat_splatinsert41:%.+]] = insertelement <8 x i32> undef, i32 [[conv40]], i32 0
// CHECK: [[splat_splat42:%.+]] = shufflevector <8 x i32> [[splat_splatinsert41]], <8 x i32> undef, <8 x i32> zeroinitializer
// CHECK: [[t28:%.+]] = load <8 x i32>, <8 x i32>* {{@.+}},
// CHECK: shl <8 x i32> [[t28]], [[splat_splat42]]
  vi8 <<= us;
// CHECK: [[t29:%.+]] = load i16, i16* @us,
// CHECK: [[conv44:%.+]] = zext i16 [[t29]] to i32
// CHECK: [[splat_splatinsert45:%.+]] = insertelement <8 x i32> undef, i32 [[conv44]], i32 0
// CHECK: [[splat_splat46:%.+]] = shufflevector <8 x i32> [[splat_splatinsert45]], <8 x i32> undef, <8 x i32> zeroinitializer
// CHECK: [[t30:%.+]] = load <8 x i32>, <8 x i32>* {{@.+}},
// CHECK: shl <8 x i32> [[t30]], [[splat_splat46]]
  vus8 <<= i;
// CHECK: [[t31:%.+]] = load i32, i32* @i,
// CHECK: [[splat_splatinsert48:%.+]] = insertelement <8 x i32> undef, i32 [[t31]], i32 0
// CHECK: [[splat_splat49:%.+]] = shufflevector <8 x i32> [[splat_splatinsert48]], <8 x i32> undef, <8 x i32> zeroinitializer
// CHECK: [[t32:%.+]] = load <8 x i16>, <8 x i16>* {{@.+}},
// CHECK: [[sh_prom50:%.+]] = trunc <8 x i32> [[splat_splat49]] to <8 x i16>
// CHECK: shl <8 x i16> [[t32]], [[sh_prom50]]
  vs8 <<= ui;
// CHECK: [[t33:%.+]] = load i32, i32* @ui,
// CHECK: [[splat_splatinsert52:%.+]] = insertelement <8 x i32> undef, i32 [[t33]], i32 0
// CHECK: [[splat_splat53:%.+]] = shufflevector <8 x i32> [[splat_splatinsert52]], <8 x i32> undef, <8 x i32> zeroinitializer
// CHECK: [[t34:%.+]] = load <8 x i16>, <8 x i16>* {{@.+}},
// CHECK: [[sh_prom54:%.+]] = trunc <8 x i32> [[splat_splat53]] to <8 x i16>
// CHECK: shl <8 x i16> [[t34]], [[sh_prom54]]
}
