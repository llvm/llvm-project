// RUN: %clang_cc1 -emit-llvm %s -o - | FileCheck %s

// PR27085

typedef unsigned char uchar4  __attribute__ ((vector_size (4)));

// CHECK: @add2
// CHECK: add <4 x i8> {{.*}}, splat (i8 2)
uchar4 add2(uchar4 v)
{
  return v + 2;
}

// CHECK: @sub2
// CHECK: sub <4 x i8> {{.*}}, splat (i8 2)
uchar4 sub2(uchar4 v)
{
  return v - 2;
}

// CHECK: @mul2
// CHECK: mul <4 x i8> {{.*}}, splat (i8 2)
uchar4 mul2(uchar4 v)
{
  return v * 2;
}

// CHECK: @div2
// CHECK: udiv <4 x i8> {{.*}}, splat (i8 2)
uchar4 div2(uchar4 v)
{
  return v / 2;
}

typedef __attribute__(( ext_vector_type(4) )) unsigned char uchar4_ext;

// CHECK: @div3_ext
// CHECK: udiv <4 x i8> %{{.*}}, splat (i8 3)
uchar4_ext div3_ext(uchar4_ext v)
{
  return v / 3;
}
