// RUN: %clang_cc1 %s -O0 -triple=x86_64-apple-darwin -target-cpu skylake-avx512 -emit-llvm -o - -Wall -Werror |opt -instnamer -S |FileCheck %s
// This test checks validity of att\gcc style inline assmebly for avx512 k and Yk constraints.
// Also checks mask register allows flexible type (size <= 64 bit)

void mask_Yk_i8(char msk){ 
//CHECK: vpaddb\09 %xmm1, %xmm0, %xmm1 {$0}\09
 asm ("vpaddb\t %%xmm1, %%xmm0, %%xmm1 %{%0%}\t"
       :                //output
       : "Yk" (msk));   //inputs
}

void mask_Yk_i16(short msk){
//CHECK: vpaddb\09 %xmm1, %xmm0, %xmm1 {$0}\09
 asm ("vpaddb\t %%xmm1, %%xmm0, %%xmm1 %{%0%}\t"
       :                //output
       :  "Yk" (msk));  //inputs
}

void mask_Yk_i32(int msk){
//CHECK: vpaddb\09 %xmm1, %xmm0, %xmm1 {$0}\09
    asm ("vpaddb\t %%xmm1, %%xmm0, %%xmm1 %{%0%}\t"
       :                //output
       :  "Yk" (msk));   //inputs
}

void mask_Yk_i64(long long msk){
//CHECK: vpaddb\09 %xmm1, %xmm0, %xmm1 {$0}\09
 asm ("vpaddb\t %%xmm1, %%xmm0, %%xmm1 %{%0%}\t"
       :                //output
       :  "Yk" (msk));   //inputs
}

void k_wise_op_i8(char msk_dst,char msk_src1,char msk_src2){
//CHECK: kandw\09$2, $1, $0
 asm ("kandw\t%2, %1, %0"
       : "=k" (msk_dst)
       : "k" (msk_src1), "k" (msk_src2));
}

void k_wise_op_i16(short msk_dst, short msk_src1, short msk_src2){
//CHECK: kandw\09$2, $1, $0
  asm ("kandw\t%2, %1, %0"
       : "=k" (msk_dst)
       : "k" (msk_src1), "k" (msk_src2));
}

void k_wise_op_i32(int msk_dst, int msk_src1, int msk_src2){
//CHECK: kandw\09$2, $1, $0
  asm ("kandw\t%2, %1, %0"
       : "=k" (msk_dst)
       : "k" (msk_src1), "k" (msk_src2));
}

void k_wise_op_i64(long long msk_dst, long long msk_src1, long long msk_src2){
//CHECK: kandw\09$2, $1, $0
  asm ("kandw\t%2, %1, %0"
       : "=k" (msk_dst)
       : "k" (msk_src1), "k" (msk_src2));
}
