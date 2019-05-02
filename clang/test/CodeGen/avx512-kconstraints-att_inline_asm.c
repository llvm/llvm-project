// RUN: %clang_cc1 %s -O0 -ffreestanding -triple=x86_64-apple-darwin -target-cpu skylake-avx512 -emit-llvm -o - -Wall -Werror |opt -instnamer -S |FileCheck %s
// This test checks validity of att\gcc style inline assmebly for avx512 k and Yk constraints.
// Also checks mask register allows flexible type (size <= 64 bit)

#include <x86intrin.h>

__m512i mask_Yk_i8(char msk, __m512i x, __m512i y){
// CHECK: <8 x i64> asm "vpaddq\09$3, $2, $0 {$1}", "=x,^Yk,x,x,~{dirflag},~{fpsr},~{flags}"(i8 %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}})
  __m512i dst;
  asm ("vpaddq\t%3, %2, %0 %{%1%}"
       : "=x" (dst)      //output
       : "Yk" (msk), "x" (x), "x" (y));   //inputs
  return dst;
}

__m512i mask_Yk_i16(short msk, __m512i x, __m512i y){
// CHECK: <8 x i64> asm "vpaddd\09$3, $2, $0 {$1}", "=x,^Yk,x,x,~{dirflag},~{fpsr},~{flags}"(i16 %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}})
  __m512i dst;
  asm ("vpaddd\t%3, %2, %0 %{%1%}"
       : "=x" (dst)      //output
       : "Yk" (msk), "x" (x), "x" (y));   //inputs
  return dst;
}

__m512i mask_Yk_i32(int msk, __m512i x, __m512i y){
// CHECK: <8 x i64> asm "vpaddw\09$3, $2, $0 {$1}", "=x,^Yk,x,x,~{dirflag},~{fpsr},~{flags}"(i32 %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}})
  __m512i dst;
  asm ("vpaddw\t%3, %2, %0 %{%1%}"
       : "=x" (dst)      //output
       : "Yk" (msk), "x" (x), "x" (y));   //inputs
  return dst;
}

__m512i mask_Yk_i64(long long msk, __m512i x, __m512i y){
// CHECK: <8 x i64> asm "vpaddb\09$3, $2, $0 {$1}", "=x,^Yk,x,x,~{dirflag},~{fpsr},~{flags}"(i64 %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}})
  __m512i dst;
  asm ("vpaddb\t%3, %2, %0 %{%1%}"
       : "=x" (dst)      //output
       : "Yk" (msk), "x" (x), "x" (y));   //inputs
  return dst;
}

char k_wise_op_i8(char msk_src1,char msk_src2){
//CHECK: i8 asm "kandb\09$2, $1, $0", "=k,k,k,~{dirflag},~{fpsr},~{flags}"(i8 %{{.*}}, i8 %{{.*}})
  char msk_dst;
  asm ("kandb\t%2, %1, %0"
       : "=k" (msk_dst)
       : "k" (msk_src1), "k" (msk_src2));
  return msk_dst;
}

short k_wise_op_i16(short msk_src1, short msk_src2){
//CHECK: i16 asm "kandw\09$2, $1, $0", "=k,k,k,~{dirflag},~{fpsr},~{flags}"(i16 %{{.*}}, i16 %{{.*}})
  short msk_dst;
  asm ("kandw\t%2, %1, %0"
       : "=k" (msk_dst)
       : "k" (msk_src1), "k" (msk_src2));
  return msk_dst;
}

int k_wise_op_i32(int msk_src1, int msk_src2){
//CHECK: i32 asm "kandd\09$2, $1, $0", "=k,k,k,~{dirflag},~{fpsr},~{flags}"(i32 %{{.*}}, i32 %{{.*}})
  int msk_dst;
  asm ("kandd\t%2, %1, %0"
       : "=k" (msk_dst)
       : "k" (msk_src1), "k" (msk_src2));
  return msk_dst;
}

long long k_wise_op_i64(long long msk_src1, long long msk_src2){
//CHECK: i64 asm "kandq\09$2, $1, $0", "=k,k,k,~{dirflag},~{fpsr},~{flags}"(i64 %{{.*}}, i64 %{{.*}})
  long long msk_dst;
  asm ("kandq\t%2, %1, %0"
       : "=k" (msk_dst)
       : "k" (msk_src1), "k" (msk_src2));
  return msk_dst;
}
