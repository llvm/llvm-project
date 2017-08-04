// FIXME: We should not be testing with -O2 (ie, a dependency on the entire IR optimizer).

// RUN: %clang_cc1 -ffreestanding %s -O2 -triple=x86_64-apple-darwin -target-cpu skylake-avx512 -emit-llvm -o - -Wall -Werror |opt -instnamer -S |FileCheck %s

#include <immintrin.h>

long long test_mm512_reduce_max_epi64(__m512i __W){
  // CHECK: %shuffle1.i = shufflevector <8 x i64> %__W, <8 x i64> undef, <8 x i32> <i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef>
  // CHECK: %tmp = icmp slt <8 x i64> %shuffle1.i, %__W
  // CHECK: %tmp1 = select <8 x i1> %tmp, <8 x i64> %__W, <8 x i64> %shuffle1.i
  // CHECK: %shuffle3.i = shufflevector <8 x i64> %tmp1, <8 x i64> undef, <8 x i32> <i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  // CHECK: %tmp2 = icmp sgt <8 x i64> %tmp1, %shuffle3.i
  // CHECK: %tmp3 = select <8 x i1> %tmp2, <8 x i64> %tmp1, <8 x i64> %shuffle3.i
  // CHECK: %shuffle6.i = shufflevector <8 x i64> %tmp3, <8 x i64> undef, <8 x i32> <i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  // CHECK: %tmp4 = icmp sgt <8 x i64> %tmp3, %shuffle6.i
  // CHECK: %.elt.i = extractelement <8 x i1> %tmp4, i32 0
  // CHECK: %.elt20.i = extractelement <8 x i64> %tmp3, i32 0
  // CHECK: %shuffle6.elt.i = extractelement <8 x i64> %tmp3, i32 1
  // CHECK: %vecext.i = select i1 %.elt.i, i64 %.elt20.i, i64 %shuffle6.elt.i
  // CHECK: ret i64 %vecext.i
  return _mm512_reduce_max_epi64(__W);
}

unsigned long long test_mm512_reduce_max_epu64(__m512i __W){
  // CHECK: %shuffle1.i = shufflevector <8 x i64> %__W, <8 x i64> undef, <8 x i32> <i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef>
  // CHECK: %tmp = icmp ult <8 x i64> %shuffle1.i, %__W
  // CHECK: %tmp1 = select <8 x i1> %tmp, <8 x i64> %__W, <8 x i64> %shuffle1.i
  // CHECK: %shuffle3.i = shufflevector <8 x i64> %tmp1, <8 x i64> undef, <8 x i32> <i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  // CHECK: %tmp2 = icmp ugt <8 x i64> %tmp1, %shuffle3.i
  // CHECK: %tmp3 = select <8 x i1> %tmp2, <8 x i64> %tmp1, <8 x i64> %shuffle3.i
  // CHECK: %shuffle6.i = shufflevector <8 x i64> %tmp3, <8 x i64> undef, <8 x i32> <i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  // CHECK: %tmp4 = icmp ugt <8 x i64> %tmp3, %shuffle6.i
  // CHECK: %.elt.i = extractelement <8 x i1> %tmp4, i32 0
  // CHECK: %.elt20.i = extractelement <8 x i64> %tmp3, i32 0
  // CHECK: %shuffle6.elt.i = extractelement <8 x i64> %tmp3, i32 1
  // CHECK: %vecext.i = select i1 %.elt.i, i64 %.elt20.i, i64 %shuffle6.elt.i
  // CHECK: ret i64 %vecext.i
  return _mm512_reduce_max_epu64(__W); 
}

double test_mm512_reduce_max_pd(__m512d __W){
  // CHECK: %shuffle1.i = shufflevector <8 x double> %__W, <8 x double> undef, <8 x i32> <i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef>
  // CHECK: %tmp = tail call <8 x double> @llvm.x86.avx512.mask.max.pd.512(<8 x double> %__W, <8 x double> %shuffle1.i, <8 x double> zeroinitializer, i8 -1, i32 4) #3
  // CHECK: %shuffle3.i = shufflevector <8 x double> %tmp, <8 x double> undef, <8 x i32> <i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  // CHECK: %tmp1 = tail call <8 x double> @llvm.x86.avx512.mask.max.pd.512(<8 x double> %tmp, <8 x double> %shuffle3.i, <8 x double> zeroinitializer, i8 -1, i32 4) #3
  // CHECK: %shuffle6.i = shufflevector <8 x double> %tmp1, <8 x double> undef, <8 x i32> <i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  // CHECK: %tmp2 = tail call <8 x double> @llvm.x86.avx512.mask.max.pd.512(<8 x double> %tmp1, <8 x double> %shuffle6.i, <8 x double> zeroinitializer, i8 -1, i32 4) #3
  // CHECK: %vecext.i = extractelement <8 x double> %tmp2, i32 0
  // CHECK: ret double %vecext.i
  return _mm512_reduce_max_pd(__W); 
}

long long test_mm512_reduce_min_epi64(__m512i __W){
  // CHECK: %shuffle1.i = shufflevector <8 x i64> %__W, <8 x i64> undef, <8 x i32> <i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef>
  // CHECK: %tmp = icmp slt <8 x i64> %shuffle1.i, %__W
  // CHECK: %tmp1 = select <8 x i1> %tmp, <8 x i64> %__W, <8 x i64> %shuffle1.i
  // CHECK: %shuffle3.i = shufflevector <8 x i64> %tmp1, <8 x i64> undef, <8 x i32> <i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  // CHECK: %tmp2 = icmp sgt <8 x i64> %tmp1, %shuffle3.i
  // CHECK: %tmp3 = select <8 x i1> %tmp2, <8 x i64> %tmp1, <8 x i64> %shuffle3.i
  // CHECK: %shuffle6.i = shufflevector <8 x i64> %tmp3, <8 x i64> undef, <8 x i32> <i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  // CHECK: %tmp4 = icmp sgt <8 x i64> %tmp3, %shuffle6.i
  // CHECK: %.elt.i = extractelement <8 x i1> %tmp4, i32 0
  // CHECK: %.elt20.i = extractelement <8 x i64> %tmp3, i32 0
  // CHECK: %shuffle6.elt.i = extractelement <8 x i64> %tmp3, i32 1
  // CHECK: %vecext.i = select i1 %.elt.i, i64 %.elt20.i, i64 %shuffle6.elt.i
  // CHECK: ret i64 %vecext.i
  return _mm512_reduce_max_epi64(__W);
}

unsigned long long test_mm512_reduce_min_epu64(__m512i __W){
  // CHECK: %shuffle1.i = shufflevector <8 x i64> %__W, <8 x i64> undef, <8 x i32> <i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef>
  // CHECK: %tmp = icmp ult <8 x i64> %shuffle1.i, %__W
  // CHECK: %tmp1 = select <8 x i1> %tmp, <8 x i64> %__W, <8 x i64> %shuffle1.i
  // CHECK: %shuffle3.i = shufflevector <8 x i64> %tmp1, <8 x i64> undef, <8 x i32> <i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  // CHECK: %tmp2 = icmp ugt <8 x i64> %tmp1, %shuffle3.i
  // CHECK: %tmp3 = select <8 x i1> %tmp2, <8 x i64> %tmp1, <8 x i64> %shuffle3.i
  // CHECK: %shuffle6.i = shufflevector <8 x i64> %tmp3, <8 x i64> undef, <8 x i32> <i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  // CHECK: %tmp4 = icmp ugt <8 x i64> %tmp3, %shuffle6.i
  // CHECK: %.elt.i = extractelement <8 x i1> %tmp4, i32 0
  // CHECK: %.elt20.i = extractelement <8 x i64> %tmp3, i32 0
  // CHECK: %shuffle6.elt.i = extractelement <8 x i64> %tmp3, i32 1
  // CHECK: %vecext.i = select i1 %.elt.i, i64 %.elt20.i, i64 %shuffle6.elt.i
  // CHECK: ret i64 %vecext.i
  return _mm512_reduce_max_epu64(__W); 
}

double test_mm512_reduce_min_pd(__m512d __W){
  // CHECK: %shuffle1.i = shufflevector <8 x double> %__W, <8 x double> undef, <8 x i32> <i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef>
  // CHECK: %tmp = tail call <8 x double> @llvm.x86.avx512.mask.min.pd.512(<8 x double> %__W, <8 x double> %shuffle1.i, <8 x double> zeroinitializer, i8 -1, i32 4) #3
  // CHECK: %shuffle3.i = shufflevector <8 x double> %tmp, <8 x double> undef, <8 x i32> <i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  // CHECK: %tmp1 = tail call <8 x double> @llvm.x86.avx512.mask.min.pd.512(<8 x double> %tmp, <8 x double> %shuffle3.i, <8 x double> zeroinitializer, i8 -1, i32 4) #3
  // CHECK: %shuffle6.i = shufflevector <8 x double> %tmp1, <8 x double> undef, <8 x i32> <i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  // CHECK: %tmp2 = tail call <8 x double> @llvm.x86.avx512.mask.min.pd.512(<8 x double> %tmp1, <8 x double> %shuffle6.i, <8 x double> zeroinitializer, i8 -1, i32 4) #3
  // CHECK: %vecext.i = extractelement <8 x double> %tmp2, i32 0
  // CHECK: ret double %vecext.i
  return _mm512_reduce_min_pd(__W); 
}

long long test_mm512_mask_reduce_max_epi64(__mmask8 __M, __m512i __W){
  // CHECK: %tmp = bitcast i8 %__M to <8 x i1>
  // CHECK: %tmp1 = select <8 x i1> %tmp, <8 x i64> %__W, <8 x i64> <i64 -9223372036854775808, i64 -9223372036854775808, i64 -9223372036854775808, i64 -9223372036854775808, i64 -9223372036854775808, i64 -9223372036854775808, i64 -9223372036854775808, i64 -9223372036854775808>
  // CHECK: %shuffle1.i = shufflevector <8 x i64> %tmp1, <8 x i64> undef, <8 x i32> <i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef>
  // CHECK: %tmp2 = icmp sgt <8 x i64> %tmp1, %shuffle1.i
  // CHECK: %tmp3 = select <8 x i1> %tmp2, <8 x i64> %tmp1, <8 x i64> %shuffle1.i
  // CHECK: %shuffle4.i = shufflevector <8 x i64> %tmp3, <8 x i64> undef, <8 x i32> <i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  // CHECK: %tmp4 = icmp sgt <8 x i64> %tmp3, %shuffle4.i
  // CHECK: %tmp5 = select <8 x i1> %tmp4, <8 x i64> %tmp3, <8 x i64> %shuffle4.i
  // CHECK: %shuffle7.i = shufflevector <8 x i64> %tmp5, <8 x i64> undef, <8 x i32> <i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  // CHECK: %tmp6 = icmp sgt <8 x i64> %tmp5, %shuffle7.i
  // CHECK: %.elt.i = extractelement <8 x i1> %tmp6, i32 0
  // CHECK: %.elt22.i = extractelement <8 x i64> %tmp5, i32 0
  // CHECK: %shuffle7.elt.i = extractelement <8 x i64> %tmp5, i32 1
  // CHECK: %vecext.i = select i1 %.elt.i, i64 %.elt22.i, i64 %shuffle7.elt.i
  // CHECK: ret i64 %vecext.i
  return _mm512_mask_reduce_max_epi64(__M, __W); 
}

unsigned long test_mm512_mask_reduce_max_epu64(__mmask8 __M, __m512i __W){
  // CHECK: %tmp = bitcast i8 %__M to <8 x i1>
  // CHECK: %tmp1 = select <8 x i1> %tmp, <8 x i64> %__W, <8 x i64> zeroinitializer
  // CHECK: %shuffle1.i = shufflevector <8 x i64> %tmp1, <8 x i64> undef, <8 x i32> <i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef>
  // CHECK: %tmp2 = icmp ugt <8 x i64> %tmp1, %shuffle1.i
  // CHECK: %tmp3 = select <8 x i1> %tmp2, <8 x i64> %tmp1, <8 x i64> %shuffle1.i
  // CHECK: %shuffle4.i = shufflevector <8 x i64> %tmp3, <8 x i64> undef, <8 x i32> <i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  // CHECK: %tmp4 = icmp ugt <8 x i64> %tmp3, %shuffle4.i
  // CHECK: %tmp5 = select <8 x i1> %tmp4, <8 x i64> %tmp3, <8 x i64> %shuffle4.i
  // CHECK: %shuffle7.i = shufflevector <8 x i64> %tmp5, <8 x i64> undef, <8 x i32> <i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  // CHECK: %tmp6 = icmp ugt <8 x i64> %tmp5, %shuffle7.i
  // CHECK: %.elt.i = extractelement <8 x i1> %tmp6, i32 0
  // CHECK: %.elt22.i = extractelement <8 x i64> %tmp5, i32 0
  // CHECK: %shuffle7.elt.i = extractelement <8 x i64> %tmp5, i32 1
  // CHECK: %vecext.i = select i1 %.elt.i, i64 %.elt22.i, i64 %shuffle7.elt.i
  // CHECK: ret i64 %vecext.i
  return _mm512_mask_reduce_max_epu64(__M, __W); 
}

long long test_mm512_mask_reduce_max_pd(__mmask8 __M, __m512d __W){
  // CHECK: %tmp = bitcast i8 %__M to <8 x i1>
  // CHECK: %tmp1 = select <8 x i1> %tmp, <8 x double> %__W, <8 x double> <double 0xFFF0000000000000, double 0xFFF0000000000000, double 0xFFF0000000000000, double 0xFFF0000000000000, double 0xFFF0000000000000, double 0xFFF0000000000000, double 0xFFF0000000000000, double 0xFFF0000000000000>
  // CHECK: %shuffle1.i = shufflevector <8 x double> %tmp1, <8 x double> undef, <8 x i32> <i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef>
  // CHECK: %tmp2 = tail call <8 x double> @llvm.x86.avx512.mask.max.pd.512(<8 x double> %tmp1, <8 x double> %shuffle1.i, <8 x double> zeroinitializer, i8 -1, i32 4) #3
  // CHECK: %shuffle4.i = shufflevector <8 x double> %tmp2, <8 x double> undef, <8 x i32> <i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  // CHECK: %tmp3 = tail call <8 x double> @llvm.x86.avx512.mask.max.pd.512(<8 x double> %tmp2, <8 x double> %shuffle4.i, <8 x double> zeroinitializer, i8 -1, i32 4) #3
  // CHECK: %shuffle7.i = shufflevector <8 x double> %tmp3, <8 x double> undef, <8 x i32> <i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  // CHECK: %tmp4 = tail call <8 x double> @llvm.x86.avx512.mask.max.pd.512(<8 x double> %tmp3, <8 x double> %shuffle7.i, <8 x double> zeroinitializer, i8 -1, i32 4) #3
  // CHECK: %vecext.i = extractelement <8 x double> %tmp4, i32 0
  // CHECK: %conv = fptosi double %vecext.i to i64
  // CHECK: ret i64 %conv
  return _mm512_mask_reduce_max_pd(__M, __W); 
}

long long test_mm512_mask_reduce_min_epi64(__mmask8 __M, __m512i __W){
  // CHECK: %tmp = bitcast i8 %__M to <8 x i1>
  // CHECK: %tmp1 = select <8 x i1> %tmp, <8 x i64> %__W, <8 x i64> <i64 9223372036854775807, i64 9223372036854775807, i64 9223372036854775807, i64 9223372036854775807, i64 9223372036854775807, i64 9223372036854775807, i64 9223372036854775807, i64 9223372036854775807>
  // CHECK: %shuffle1.i = shufflevector <8 x i64> %tmp1, <8 x i64> undef, <8 x i32> <i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef>
  // CHECK: %tmp2 = icmp slt <8 x i64> %tmp1, %shuffle1.i
  // CHECK: %tmp3 = select <8 x i1> %tmp2, <8 x i64> %tmp1, <8 x i64> %shuffle1.i
  // CHECK: %shuffle4.i = shufflevector <8 x i64> %tmp3, <8 x i64> undef, <8 x i32> <i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  // CHECK: %tmp4 = icmp slt <8 x i64> %tmp3, %shuffle4.i
  // CHECK: %tmp5 = select <8 x i1> %tmp4, <8 x i64> %tmp3, <8 x i64> %shuffle4.i
  // CHECK: %shuffle7.i = shufflevector <8 x i64> %tmp5, <8 x i64> undef, <8 x i32> <i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  // CHECK: %tmp6 = icmp slt <8 x i64> %tmp5, %shuffle7.i
  // CHECK: %.elt.i = extractelement <8 x i1> %tmp6, i32 0
  // CHECK: %.elt22.i = extractelement <8 x i64> %tmp5, i32 0
  // CHECK: %shuffle7.elt.i = extractelement <8 x i64> %tmp5, i32 1
  // CHECK: %vecext.i = select i1 %.elt.i, i64 %.elt22.i, i64 %shuffle7.elt.i
  // CHECK: ret i64 %vecext.i
  return _mm512_mask_reduce_min_epi64(__M, __W); 
}

long long test_mm512_mask_reduce_min_epu64(__mmask8 __M, __m512i __W){
  // CHECK: %tmp = bitcast i8 %__M to <8 x i1>
  // CHECK: %tmp1 = select <8 x i1> %tmp, <8 x i64> %__W, <8 x i64> zeroinitializer
  // CHECK: %shuffle1.i = shufflevector <8 x i64> %tmp1, <8 x i64> undef, <8 x i32> <i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef>
  // CHECK: %tmp2 = icmp ugt <8 x i64> %tmp1, %shuffle1.i
  // CHECK: %tmp3 = select <8 x i1> %tmp2, <8 x i64> %tmp1, <8 x i64> %shuffle1.i
  // CHECK: %shuffle4.i = shufflevector <8 x i64> %tmp3, <8 x i64> undef, <8 x i32> <i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  // CHECK: %tmp4 = icmp ugt <8 x i64> %tmp3, %shuffle4.i
  // CHECK: %tmp5 = select <8 x i1> %tmp4, <8 x i64> %tmp3, <8 x i64> %shuffle4.i
  // CHECK: %shuffle7.i = shufflevector <8 x i64> %tmp5, <8 x i64> undef, <8 x i32> <i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  // CHECK: %tmp6 = icmp ugt <8 x i64> %tmp5, %shuffle7.i
  // CHECK: %.elt.i = extractelement <8 x i1> %tmp6, i32 0
  // CHECK: %.elt22.i = extractelement <8 x i64> %tmp5, i32 0
  // CHECK: %shuffle7.elt.i = extractelement <8 x i64> %tmp5, i32 1
  // CHECK: %vecext.i = select i1 %.elt.i, i64 %.elt22.i, i64 %shuffle7.elt.i
  // CHECK: ret i64 %vecext.i
  return _mm512_mask_reduce_max_epu64(__M, __W); 
}

double test_mm512_mask_reduce_min_pd(__mmask8 __M, __m512d __W){
  // CHECK: %tmp = bitcast i8 %__M to <8 x i1>
  // CHECK: %tmp1 = select <8 x i1> %tmp, <8 x double> %__W, <8 x double> <double 0x7FF0000000000000, double 0x7FF0000000000000, double 0x7FF0000000000000, double 0x7FF0000000000000, double 0x7FF0000000000000, double 0x7FF0000000000000, double 0x7FF0000000000000, double 0x7FF0000000000000>
  // CHECK: %shuffle1.i = shufflevector <8 x double> %tmp1, <8 x double> undef, <8 x i32> <i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef>
  // CHECK: %tmp2 = tail call <8 x double> @llvm.x86.avx512.mask.min.pd.512(<8 x double> %tmp1, <8 x double> %shuffle1.i, <8 x double> zeroinitializer, i8 -1, i32 4) #3
  // CHECK: %shuffle4.i = shufflevector <8 x double> %tmp2, <8 x double> undef, <8 x i32> <i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  // CHECK: %tmp3 = tail call <8 x double> @llvm.x86.avx512.mask.min.pd.512(<8 x double> %tmp2, <8 x double> %shuffle4.i, <8 x double> zeroinitializer, i8 -1, i32 4) #3
  // CHECK: %shuffle7.i = shufflevector <8 x double> %tmp3, <8 x double> undef, <8 x i32> <i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  // CHECK: %tmp4 = tail call <8 x double> @llvm.x86.avx512.mask.min.pd.512(<8 x double> %tmp3, <8 x double> %shuffle7.i, <8 x double> zeroinitializer, i8 -1, i32 4) #3
  // CHECK: %vecext.i = extractelement <8 x double> %tmp4, i32 0
  // CHECK: ret double %vecext.i
  return _mm512_mask_reduce_min_pd(__M, __W); 
}

int test_mm512_reduce_max_epi32(__m512i __W){
  // CHECK: %tmp = bitcast <8 x i64> %__W to <16 x i32>
  // CHECK: %shuffle1.i = shufflevector <16 x i32> %tmp, <16 x i32> undef, <16 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  // CHECK: %tmp1 = icmp slt <16 x i32> %shuffle1.i, %tmp
  // CHECK: %tmp2 = select <16 x i1> %tmp1, <16 x i32> %tmp, <16 x i32> %shuffle1.i
  // CHECK: %shuffle3.i = shufflevector <16 x i32> %tmp2, <16 x i32> undef, <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  // CHECK: %tmp3 = icmp sgt <16 x i32> %tmp2, %shuffle3.i
  // CHECK: %tmp4 = select <16 x i1> %tmp3, <16 x i32> %tmp2, <16 x i32> %shuffle3.i
  // CHECK: %shuffle6.i = shufflevector <16 x i32> %tmp4, <16 x i32> undef, <16 x i32> <i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  // CHECK: %tmp5 = icmp sgt <16 x i32> %tmp4, %shuffle6.i
  // CHECK: %tmp6 = select <16 x i1> %tmp5, <16 x i32> %tmp4, <16 x i32> %shuffle6.i
  // CHECK: %shuffle9.i = shufflevector <16 x i32> %tmp6, <16 x i32> undef, <16 x i32> <i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  // CHECK: %tmp7 = icmp sgt <16 x i32> %tmp6, %shuffle9.i
  // CHECK: %tmp8 = select <16 x i1> %tmp7, <16 x i32> %tmp6, <16 x i32> %shuffle9.i
  // CHECK: %tmp9 = bitcast <16 x i32> %tmp8 to <8 x i64>
  // CHECK: %vecext.i = extractelement <8 x i64> %tmp9, i32 0
  // CHECK: %conv.i = trunc i64 %vecext.i to i32
  // CHECK: ret i32 %conv.i
  return _mm512_reduce_max_epi32(__W);
}

unsigned int test_mm512_reduce_max_epu32(__m512i __W){
  // CHECK: %tmp = bitcast <8 x i64> %__W to <16 x i32>
  // CHECK: %shuffle1.i = shufflevector <16 x i32> %tmp, <16 x i32> undef, <16 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  // CHECK: %tmp1 = icmp ult <16 x i32> %shuffle1.i, %tmp
  // CHECK: %tmp2 = select <16 x i1> %tmp1, <16 x i32> %tmp, <16 x i32> %shuffle1.i
  // CHECK: %shuffle3.i = shufflevector <16 x i32> %tmp2, <16 x i32> undef, <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  // CHECK: %tmp3 = icmp ugt <16 x i32> %tmp2, %shuffle3.i
  // CHECK: %tmp4 = select <16 x i1> %tmp3, <16 x i32> %tmp2, <16 x i32> %shuffle3.i
  // CHECK: %shuffle6.i = shufflevector <16 x i32> %tmp4, <16 x i32> undef, <16 x i32> <i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  // CHECK: %tmp5 = icmp ugt <16 x i32> %tmp4, %shuffle6.i
  // CHECK: %tmp6 = select <16 x i1> %tmp5, <16 x i32> %tmp4, <16 x i32> %shuffle6.i
  // CHECK: %shuffle9.i = shufflevector <16 x i32> %tmp6, <16 x i32> undef, <16 x i32> <i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  // CHECK: %tmp7 = icmp ugt <16 x i32> %tmp6, %shuffle9.i
  // CHECK: %tmp8 = select <16 x i1> %tmp7, <16 x i32> %tmp6, <16 x i32> %shuffle9.i
  // CHECK: %tmp9 = bitcast <16 x i32> %tmp8 to <8 x i64>
  // CHECK: %vecext.i = extractelement <8 x i64> %tmp9, i32 0
  // CHECK: %conv.i = trunc i64 %vecext.i to i32
  // CHECK: ret i32 %conv.i
  return _mm512_reduce_max_epu32(__W); 
}

float test_mm512_reduce_max_ps(__m512 __W){
  // CHECK: %shuffle1.i = shufflevector <16 x float> %__W, <16 x float> undef, <16 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  // CHECK: %tmp = tail call <16 x float> @llvm.x86.avx512.mask.max.ps.512(<16 x float> %__W, <16 x float> %shuffle1.i, <16 x float> zeroinitializer, i16 -1, i32 4) #3
  // CHECK: %shuffle3.i = shufflevector <16 x float> %tmp, <16 x float> undef, <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  // CHECK: %tmp1 = tail call <16 x float> @llvm.x86.avx512.mask.max.ps.512(<16 x float> %tmp, <16 x float> %shuffle3.i, <16 x float> zeroinitializer, i16 -1, i32 4) #3
  // CHECK: %shuffle6.i = shufflevector <16 x float> %tmp1, <16 x float> undef, <16 x i32> <i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  // CHECK: %tmp2 = tail call <16 x float> @llvm.x86.avx512.mask.max.ps.512(<16 x float> %tmp1, <16 x float> %shuffle6.i, <16 x float> zeroinitializer, i16 -1, i32 4) #3
  // CHECK: %shuffle9.i = shufflevector <16 x float> %tmp2, <16 x float> undef, <16 x i32> <i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  // CHECK: %tmp3 = tail call <16 x float> @llvm.x86.avx512.mask.max.ps.512(<16 x float> %tmp2, <16 x float> %shuffle9.i, <16 x float> zeroinitializer, i16 -1, i32 4) #3
  // CHECK: %vecext.i = extractelement <16 x float> %tmp3, i32 0
  // CHECK: ret float %vecext.i
  return _mm512_reduce_max_ps(__W); 
}

int test_mm512_reduce_min_epi32(__m512i __W){
  // CHECK: %tmp = bitcast <8 x i64> %__W to <16 x i32>
  // CHECK: %shuffle1.i = shufflevector <16 x i32> %tmp, <16 x i32> undef, <16 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  // CHECK: %tmp1 = icmp sgt <16 x i32> %shuffle1.i, %tmp
  // CHECK: %tmp2 = select <16 x i1> %tmp1, <16 x i32> %tmp, <16 x i32> %shuffle1.i
  // CHECK: %shuffle3.i = shufflevector <16 x i32> %tmp2, <16 x i32> undef, <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  // CHECK: %tmp3 = icmp slt <16 x i32> %tmp2, %shuffle3.i
  // CHECK: %tmp4 = select <16 x i1> %tmp3, <16 x i32> %tmp2, <16 x i32> %shuffle3.i
  // CHECK: %shuffle6.i = shufflevector <16 x i32> %tmp4, <16 x i32> undef, <16 x i32> <i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  // CHECK: %tmp5 = icmp slt <16 x i32> %tmp4, %shuffle6.i
  // CHECK: %tmp6 = select <16 x i1> %tmp5, <16 x i32> %tmp4, <16 x i32> %shuffle6.i
  // CHECK: %shuffle9.i = shufflevector <16 x i32> %tmp6, <16 x i32> undef, <16 x i32> <i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  // CHECK: %tmp7 = icmp slt <16 x i32> %tmp6, %shuffle9.i
  // CHECK: %tmp8 = select <16 x i1> %tmp7, <16 x i32> %tmp6, <16 x i32> %shuffle9.i
  // CHECK: %tmp9 = bitcast <16 x i32> %tmp8 to <8 x i64>
  // CHECK: %vecext.i = extractelement <8 x i64> %tmp9, i32 0
  // CHECK: %conv.i = trunc i64 %vecext.i to i32
  // CHECK: ret i32 %conv.i
  return _mm512_reduce_min_epi32(__W);
}

unsigned int test_mm512_reduce_min_epu32(__m512i __W){
  // CHECK: %tmp = bitcast <8 x i64> %__W to <16 x i32>
  // CHECK: %shuffle1.i = shufflevector <16 x i32> %tmp, <16 x i32> undef, <16 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  // CHECK: %tmp1 = icmp ugt <16 x i32> %shuffle1.i, %tmp
  // CHECK: %tmp2 = select <16 x i1> %tmp1, <16 x i32> %tmp, <16 x i32> %shuffle1.i
  // CHECK: %shuffle3.i = shufflevector <16 x i32> %tmp2, <16 x i32> undef, <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  // CHECK: %tmp3 = icmp ult <16 x i32> %tmp2, %shuffle3.i
  // CHECK: %tmp4 = select <16 x i1> %tmp3, <16 x i32> %tmp2, <16 x i32> %shuffle3.i
  // CHECK: %shuffle6.i = shufflevector <16 x i32> %tmp4, <16 x i32> undef, <16 x i32> <i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  // CHECK: %tmp5 = icmp ult <16 x i32> %tmp4, %shuffle6.i
  // CHECK: %tmp6 = select <16 x i1> %tmp5, <16 x i32> %tmp4, <16 x i32> %shuffle6.i
  // CHECK: %shuffle9.i = shufflevector <16 x i32> %tmp6, <16 x i32> undef, <16 x i32> <i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  // CHECK: %tmp7 = icmp ult <16 x i32> %tmp6, %shuffle9.i
  // CHECK: %tmp8 = select <16 x i1> %tmp7, <16 x i32> %tmp6, <16 x i32> %shuffle9.i
  // CHECK: %tmp9 = bitcast <16 x i32> %tmp8 to <8 x i64>
  // CHECK: %vecext.i = extractelement <8 x i64> %tmp9, i32 0
  // CHECK: %conv.i = trunc i64 %vecext.i to i32
  // CHECK: ret i32 %conv.i
  return _mm512_reduce_min_epu32(__W); 
}

float test_mm512_reduce_min_ps(__m512 __W){
  // CHECK: %shuffle1.i = shufflevector <16 x float> %__W, <16 x float> undef, <16 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  // CHECK: %tmp = tail call <16 x float> @llvm.x86.avx512.mask.min.ps.512(<16 x float> %__W, <16 x float> %shuffle1.i, <16 x float> zeroinitializer, i16 -1, i32 4) #3
  // CHECK: %shuffle3.i = shufflevector <16 x float> %tmp, <16 x float> undef, <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  // CHECK: %tmp1 = tail call <16 x float> @llvm.x86.avx512.mask.min.ps.512(<16 x float> %tmp, <16 x float> %shuffle3.i, <16 x float> zeroinitializer, i16 -1, i32 4) #3
  // CHECK: %shuffle6.i = shufflevector <16 x float> %tmp1, <16 x float> undef, <16 x i32> <i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  // CHECK: %tmp2 = tail call <16 x float> @llvm.x86.avx512.mask.min.ps.512(<16 x float> %tmp1, <16 x float> %shuffle6.i, <16 x float> zeroinitializer, i16 -1, i32 4) #3
  // CHECK: %shuffle9.i = shufflevector <16 x float> %tmp2, <16 x float> undef, <16 x i32> <i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  // CHECK: %tmp3 = tail call <16 x float> @llvm.x86.avx512.mask.min.ps.512(<16 x float> %tmp2, <16 x float> %shuffle9.i, <16 x float> zeroinitializer, i16 -1, i32 4) #3
  // CHECK: %vecext.i = extractelement <16 x float> %tmp3, i32 0
  // CHECK: ret float %vecext.i
  return _mm512_reduce_min_ps(__W); 
}

int test_mm512_mask_reduce_max_epi32(__mmask16 __M, __m512i __W){
  // CHECK: %tmp = bitcast <8 x i64> %__W to <16 x i32>
  // CHECK: %tmp1 = bitcast i16 %__M to <16 x i1>
  // CHECK: %tmp2 = select <16 x i1> %tmp1, <16 x i32> %tmp, <16 x i32> <i32 -2147483648, i32 -2147483648, i32 -2147483648, i32 -2147483648, i32 -2147483648, i32 -2147483648, i32 -2147483648, i32 -2147483648, i32 -2147483648, i32 -2147483648, i32 -2147483648, i32 -2147483648, i32 -2147483648, i32 -2147483648, i32 -2147483648, i32 -2147483648>
  // CHECK: %shuffle1.i = shufflevector <16 x i32> %tmp2, <16 x i32> undef, <16 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  // CHECK: %tmp3 = icmp sgt <16 x i32> %tmp2, %shuffle1.i
  // CHECK: %tmp4 = select <16 x i1> %tmp3, <16 x i32> %tmp2, <16 x i32> %shuffle1.i
  // CHECK: %shuffle4.i = shufflevector <16 x i32> %tmp4, <16 x i32> undef, <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  // CHECK: %tmp5 = icmp sgt <16 x i32> %tmp4, %shuffle4.i
  // CHECK: %tmp6 = select <16 x i1> %tmp5, <16 x i32> %tmp4, <16 x i32> %shuffle4.i
  // CHECK: %shuffle7.i = shufflevector <16 x i32> %tmp6, <16 x i32> undef, <16 x i32> <i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  // CHECK: %tmp7 = icmp sgt <16 x i32> %tmp6, %shuffle7.i
  // CHECK: %tmp8 = select <16 x i1> %tmp7, <16 x i32> %tmp6, <16 x i32> %shuffle7.i
  // CHECK: %shuffle10.i = shufflevector <16 x i32> %tmp8, <16 x i32> undef, <16 x i32> <i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  // CHECK: %tmp9 = icmp sgt <16 x i32> %tmp8, %shuffle10.i
  // CHECK: %tmp10 = select <16 x i1> %tmp9, <16 x i32> %tmp8, <16 x i32> %shuffle10.i
  // CHECK: %tmp11 = bitcast <16 x i32> %tmp10 to <8 x i64>
  // CHECK: %vecext.i = extractelement <8 x i64> %tmp11, i32 0
  // CHECK: %conv.i = trunc i64 %vecext.i to i32
  // CHECK: ret i32 %conv.i
  return _mm512_mask_reduce_max_epi32(__M, __W); 
}

unsigned int test_mm512_mask_reduce_max_epu32(__mmask16 __M, __m512i __W){
  // CHECK: %tmp = bitcast <8 x i64> %__W to <16 x i32>
  // CHECK: %tmp1 = bitcast i16 %__M to <16 x i1>
  // CHECK: %tmp2 = select <16 x i1> %tmp1, <16 x i32> %tmp, <16 x i32> zeroinitializer
  // CHECK: %shuffle1.i = shufflevector <16 x i32> %tmp2, <16 x i32> undef, <16 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  // CHECK: %tmp3 = icmp ugt <16 x i32> %tmp2, %shuffle1.i
  // CHECK: %tmp4 = select <16 x i1> %tmp3, <16 x i32> %tmp2, <16 x i32> %shuffle1.i
  // CHECK: %shuffle4.i = shufflevector <16 x i32> %tmp4, <16 x i32> undef, <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  // CHECK: %tmp5 = icmp ugt <16 x i32> %tmp4, %shuffle4.i
  // CHECK: %tmp6 = select <16 x i1> %tmp5, <16 x i32> %tmp4, <16 x i32> %shuffle4.i
  // CHECK: %shuffle7.i = shufflevector <16 x i32> %tmp6, <16 x i32> undef, <16 x i32> <i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  // CHECK: %tmp7 = icmp ugt <16 x i32> %tmp6, %shuffle7.i
  // CHECK: %tmp8 = select <16 x i1> %tmp7, <16 x i32> %tmp6, <16 x i32> %shuffle7.i
  // CHECK: %shuffle10.i = shufflevector <16 x i32> %tmp8, <16 x i32> undef, <16 x i32> <i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  // CHECK: %tmp9 = icmp ugt <16 x i32> %tmp8, %shuffle10.i
  // CHECK: %tmp10 = select <16 x i1> %tmp9, <16 x i32> %tmp8, <16 x i32> %shuffle10.i
  // CHECK: %tmp11 = bitcast <16 x i32> %tmp10 to <8 x i64>
  // CHECK: %vecext.i = extractelement <8 x i64> %tmp11, i32 0
  // CHECK: %conv.i = trunc i64 %vecext.i to i32
  // CHECK: ret i32 %conv.i
  return _mm512_mask_reduce_max_epu32(__M, __W); 
}

float test_mm512_mask_reduce_max_ps(__mmask16 __M, __m512 __W){
  // CHECK: %tmp = bitcast i16 %__M to <16 x i1>
  // CHECK: %tmp1 = select <16 x i1> %tmp, <16 x float> %__W, <16 x float> <float 0xFFF0000000000000, float 0xFFF0000000000000, float 0xFFF0000000000000, float 0xFFF0000000000000, float 0xFFF0000000000000, float 0xFFF0000000000000, float 0xFFF0000000000000, float 0xFFF0000000000000, float 0xFFF0000000000000, float 0xFFF0000000000000, float 0xFFF0000000000000, float 0xFFF0000000000000, float 0xFFF0000000000000, float 0xFFF0000000000000, float 0xFFF0000000000000, float 0xFFF0000000000000>
  // CHECK: %shuffle1.i = shufflevector <16 x float> %tmp1, <16 x float> undef, <16 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  // CHECK: %tmp2 = tail call <16 x float> @llvm.x86.avx512.mask.max.ps.512(<16 x float> %tmp1, <16 x float> %shuffle1.i, <16 x float> zeroinitializer, i16 -1, i32 4) #3
  // CHECK: %shuffle4.i = shufflevector <16 x float> %tmp2, <16 x float> undef, <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  // CHECK: %tmp3 = tail call <16 x float> @llvm.x86.avx512.mask.max.ps.512(<16 x float> %tmp2, <16 x float> %shuffle4.i, <16 x float> zeroinitializer, i16 -1, i32 4) #3
  // CHECK: %shuffle7.i = shufflevector <16 x float> %tmp3, <16 x float> undef, <16 x i32> <i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  // CHECK: %tmp4 = tail call <16 x float> @llvm.x86.avx512.mask.max.ps.512(<16 x float> %tmp3, <16 x float> %shuffle7.i, <16 x float> zeroinitializer, i16 -1, i32 4) #3
  // CHECK: %shuffle10.i = shufflevector <16 x float> %tmp4, <16 x float> undef, <16 x i32> <i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  // CHECK: %tmp5 = tail call <16 x float> @llvm.x86.avx512.mask.max.ps.512(<16 x float> %tmp4, <16 x float> %shuffle10.i, <16 x float> zeroinitializer, i16 -1, i32 4) #3
  // CHECK: %vecext.i = extractelement <16 x float> %tmp5, i32 0
  // CHECK: ret float %vecext.i
  return _mm512_mask_reduce_max_ps(__M, __W); 
}

int test_mm512_mask_reduce_min_epi32(__mmask16 __M, __m512i __W){
  // CHECK: %tmp = bitcast <8 x i64> %__W to <16 x i32>
  // CHECK: %tmp1 = bitcast i16 %__M to <16 x i1>
  // CHECK: %tmp2 = select <16 x i1> %tmp1, <16 x i32> %tmp, <16 x i32> <i32 2147483647, i32 2147483647, i32 2147483647, i32 2147483647, i32 2147483647, i32 2147483647, i32 2147483647, i32 2147483647, i32 2147483647, i32 2147483647, i32 2147483647, i32 2147483647, i32 2147483647, i32 2147483647, i32 2147483647, i32 2147483647>
  // CHECK: %shuffle1.i = shufflevector <16 x i32> %tmp2, <16 x i32> undef, <16 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  // CHECK: %tmp3 = icmp slt <16 x i32> %tmp2, %shuffle1.i
  // CHECK: %tmp4 = select <16 x i1> %tmp3, <16 x i32> %tmp2, <16 x i32> %shuffle1.i
  // CHECK: %shuffle4.i = shufflevector <16 x i32> %tmp4, <16 x i32> undef, <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  // CHECK: %tmp5 = icmp slt <16 x i32> %tmp4, %shuffle4.i
  // CHECK: %tmp6 = select <16 x i1> %tmp5, <16 x i32> %tmp4, <16 x i32> %shuffle4.i
  // CHECK: %shuffle7.i = shufflevector <16 x i32> %tmp6, <16 x i32> undef, <16 x i32> <i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  // CHECK: %tmp7 = icmp slt <16 x i32> %tmp6, %shuffle7.i
  // CHECK: %tmp8 = select <16 x i1> %tmp7, <16 x i32> %tmp6, <16 x i32> %shuffle7.i
  // CHECK: %shuffle10.i = shufflevector <16 x i32> %tmp8, <16 x i32> undef, <16 x i32> <i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  // CHECK: %tmp9 = icmp slt <16 x i32> %tmp8, %shuffle10.i
  // CHECK: %tmp10 = select <16 x i1> %tmp9, <16 x i32> %tmp8, <16 x i32> %shuffle10.i
  // CHECK: %tmp11 = bitcast <16 x i32> %tmp10 to <8 x i64>
  // CHECK: %vecext.i = extractelement <8 x i64> %tmp11, i32 0
  // CHECK: %conv.i = trunc i64 %vecext.i to i32
  // CHECK: ret i32 %conv.i
  return _mm512_mask_reduce_min_epi32(__M, __W); 
}

unsigned int test_mm512_mask_reduce_min_epu32(__mmask16 __M, __m512i __W){
  // CHECK: %tmp = bitcast <8 x i64> %__W to <16 x i32>
  // CHECK: %tmp1 = bitcast i16 %__M to <16 x i1>
  // CHECK: %tmp2 = select <16 x i1> %tmp1, <16 x i32> %tmp, <16 x i32> <i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1>
  // CHECK: %shuffle1.i = shufflevector <16 x i32> %tmp2, <16 x i32> undef, <16 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  // CHECK: %tmp3 = icmp ult <16 x i32> %tmp2, %shuffle1.i
  // CHECK: %tmp4 = select <16 x i1> %tmp3, <16 x i32> %tmp2, <16 x i32> %shuffle1.i
  // CHECK: %shuffle4.i = shufflevector <16 x i32> %tmp4, <16 x i32> undef, <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  // CHECK: %tmp5 = icmp ult <16 x i32> %tmp4, %shuffle4.i
  // CHECK: %tmp6 = select <16 x i1> %tmp5, <16 x i32> %tmp4, <16 x i32> %shuffle4.i
  // CHECK: %shuffle7.i = shufflevector <16 x i32> %tmp6, <16 x i32> undef, <16 x i32> <i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  // CHECK: %tmp7 = icmp ult <16 x i32> %tmp6, %shuffle7.i
  // CHECK: %tmp8 = select <16 x i1> %tmp7, <16 x i32> %tmp6, <16 x i32> %shuffle7.i
  // CHECK: %shuffle10.i = shufflevector <16 x i32> %tmp8, <16 x i32> undef, <16 x i32> <i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  // CHECK: %tmp9 = icmp ult <16 x i32> %tmp8, %shuffle10.i
  // CHECK: %tmp10 = select <16 x i1> %tmp9, <16 x i32> %tmp8, <16 x i32> %shuffle10.i
  // CHECK: %tmp11 = bitcast <16 x i32> %tmp10 to <8 x i64>
  // CHECK: %vecext.i = extractelement <8 x i64> %tmp11, i32 0
  // CHECK: %conv.i = trunc i64 %vecext.i to i32
  // CHECK: ret i32 %conv.i
  return _mm512_mask_reduce_min_epu32(__M, __W); 
}

float test_mm512_mask_reduce_min_ps(__mmask16 __M, __m512 __W){
  // CHECK: %tmp = bitcast i16 %__M to <16 x i1>
  // CHECK: %tmp1 = select <16 x i1> %tmp, <16 x float> %__W, <16 x float> <float 0x7FF0000000000000, float 0x7FF0000000000000, float 0x7FF0000000000000, float 0x7FF0000000000000, float 0x7FF0000000000000, float 0x7FF0000000000000, float 0x7FF0000000000000, float 0x7FF0000000000000, float 0x7FF0000000000000, float 0x7FF0000000000000, float 0x7FF0000000000000, float 0x7FF0000000000000, float 0x7FF0000000000000, float 0x7FF0000000000000, float 0x7FF0000000000000, float 0x7FF0000000000000>
  // CHECK: %shuffle1.i = shufflevector <16 x float> %tmp1, <16 x float> undef, <16 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  // CHECK: %tmp2 = tail call <16 x float> @llvm.x86.avx512.mask.min.ps.512(<16 x float> %tmp1, <16 x float> %shuffle1.i, <16 x float> zeroinitializer, i16 -1, i32 4) #3
  // CHECK: %shuffle4.i = shufflevector <16 x float> %tmp2, <16 x float> undef, <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  // CHECK: %tmp3 = tail call <16 x float> @llvm.x86.avx512.mask.min.ps.512(<16 x float> %tmp2, <16 x float> %shuffle4.i, <16 x float> zeroinitializer, i16 -1, i32 4) #3
  // CHECK: %shuffle7.i = shufflevector <16 x float> %tmp3, <16 x float> undef, <16 x i32> <i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  // CHECK: %tmp4 = tail call <16 x float> @llvm.x86.avx512.mask.min.ps.512(<16 x float> %tmp3, <16 x float> %shuffle7.i, <16 x float> zeroinitializer, i16 -1, i32 4) #3
  // CHECK: %shuffle10.i = shufflevector <16 x float> %tmp4, <16 x float> undef, <16 x i32> <i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  // CHECK: %tmp5 = tail call <16 x float> @llvm.x86.avx512.mask.min.ps.512(<16 x float> %tmp4, <16 x float> %shuffle10.i, <16 x float> zeroinitializer, i16 -1, i32 4) #3
  // CHECK: %vecext.i = extractelement <16 x float> %tmp5, i32 0
  // CHECK: ret float %vecext.i
  return _mm512_mask_reduce_min_ps(__M, __W); 
}

