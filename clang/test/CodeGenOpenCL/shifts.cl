// RUN: %clang_cc1 -x cl -O1 -emit-llvm  %s -o - -triple x86_64-linux-gnu | FileCheck %s -check-prefix=OPT
// RUN: %clang_cc1 -x cl -O0 -emit-llvm  %s -o - -triple x86_64-linux-gnu | FileCheck %s -check-prefix=NOOPT

// OpenCL essentially reduces all shift amounts to the last word-size
// bits before evaluating. Test this both for variables and constants
// evaluated in the front-end.

// OPT: @gtest1 ={{.*}} local_unnamed_addr constant i64 2147483648
__constant const unsigned long gtest1 = 1UL << 31;

// NOOPT: @negativeShift32
int negativeShift32(int a,int b) {
  // NOOPT: %array0 = alloca [256 x i8]
  char array0[((int)1)<<40];
  // NOOPT: %array1 = alloca [256 x i8]
  char array1[((int)1)<<(-24)];

  // NOOPT: ret i32 65536
  return ((int)1)<<(-16);
}

//OPT: @positiveShift32
int positiveShift32(int a,int b) {
  //OPT: [[M32:%.+]] = and i32 %b, 31
  //OPT-NEXT: [[C32:%.+]] = shl i32 %a, [[M32]]
  int c = a<<b;
  int d = ((int)1)<<33;
  //OPT-NEXT: [[E32:%.+]] = add nsw i32 [[C32]], 2
  int e = c + d;
  //OPT-NEXT: ret i32 [[E32]]
  return e;
}

//OPT: @positiveShift64
long positiveShift64(long a,long b) {
  //OPT: [[M64:%.+]] = and i64 %b, 63
  //OPT-NEXT: [[C64:%.+]] = ashr i64 %a, [[M64]]
  long c = a>>b;
  long d = ((long)8)>>65;
  //OPT-NEXT: [[E64:%.+]] = add nsw i64 [[C64]], 4
  long e = c + d;
  //OPT-NEXT: ret i64 [[E64]]
  return e;
}

typedef __attribute__((ext_vector_type(4))) int int4;

//OPT: @vectorVectorTest
int4 vectorVectorTest(int4 a,int4 b) {
  //OPT: [[VM:%.+]] = and <4 x i32> %b, <i32 31, i32 31, i32 31, i32 31>
  //OPT-NEXT: [[VC:%.+]] = shl <4 x i32> %a, [[VM]]
  int4 c = a << b;
  //OPT-NEXT: [[VF:%.+]] = add <4 x i32> [[VC]], <i32 2, i32 4, i32 16, i32 8>
  int4 d = {1, 1, 1, 1};
  int4 e = {33, 34, -28, -29};
  int4 f = c + (d << e);
  //OPT-NEXT: ret <4 x i32> [[VF]]
  return f;
}

//NOOPT-LABEL: @vectorScalarTest
int4 vectorScalarTest(int4 a,int b) {
  //NOOPT: [[SP0:%.+]] = insertelement <4 x i32> poison
  //NOOPT: [[SP1:%.+]] = shufflevector <4 x i32> [[SP0]], <4 x i32> poison, <4 x i32> zeroinitializer
  //NOOPT: [[VSM:%.+]] = and <4 x i32> [[SP1]], <i32 31, i32 31, i32 31, i32 31>
  //NOOPT: [[VSC:%.+]] = shl <4 x i32> [[VSS:%.+]], [[VSM]]
  int4 c = a << b;
  //NOOPT: [[VSF:%.+]] = shl <4 x i32> [[VSC1:%.+]], <i32 2, i32 2, i32 2, i32 2>
  //NOOPT: [[VSA:%.+]] = add <4 x i32> [[VSC2:%.+]], [[VSF]]
  int4 d = {1, 1, 1, 1};
  int4 f = c + (d << 34);
  //NOOPT: ret <4 x i32>
  return f;
}
