// RUN: %clang_cc1 -ffreestanding %s -O0 -triple=x86_64-apple-darwin -target-cpu skylake-avx512 -emit-llvm -o - -Wall -Werror | FileCheck %s

#include <immintrin.h>

// CHECK-LABEL: define i64 @test_mm512_reduce_max_epi64(<8 x i64> %__W) #0 {
// CHECK:   [[__A_ADDR_I12_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__B_ADDR_I13_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__A_ADDR_I9_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__B_ADDR_I10_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__A_ADDR_I_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__B_ADDR_I_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__V_ADDR_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__W_ADDR:%.*]] = alloca <8 x i64>, align 64
// CHECK:   store <8 x i64> %__W, <8 x i64>* [[__W_ADDR]], align 64
// CHECK:   [[TMP0:%.*]] = load <8 x i64>, <8 x i64>* [[__W_ADDR]], align 64
// CHECK:   store <8 x i64> [[TMP0]], <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP1:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP2:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i64> [[TMP1]], <8 x i64> [[TMP2]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP3:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP4:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[SHUFFLE1_I:%.*]] = shufflevector <8 x i64> [[TMP3]], <8 x i64> [[TMP4]], <8 x i32> <i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   store <8 x i64> [[SHUFFLE_I]], <8 x i64>* [[__A_ADDR_I_I]], align 64
// CHECK:   store <8 x i64> [[SHUFFLE1_I]], <8 x i64>* [[__B_ADDR_I_I]], align 64
// CHECK:   [[TMP5:%.*]] = load <8 x i64>, <8 x i64>* [[__A_ADDR_I_I]], align 64
// CHECK:   [[TMP6:%.*]] = load <8 x i64>, <8 x i64>* [[__B_ADDR_I_I]], align 64
// CHECK:   [[TMP8:%.*]] = icmp sgt <8 x i64> [[TMP5]], [[TMP6]]
// CHECK:   [[TMP9:%.*]] = select <8 x i1> [[TMP8]], <8 x i64> [[TMP5]], <8 x i64> [[TMP6]]
// CHECK:   store <8 x i64> [[TMP9]], <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP10:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP11:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[SHUFFLE2_I:%.*]] = shufflevector <8 x i64> [[TMP10]], <8 x i64> [[TMP11]], <8 x i32> <i32 0, i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP12:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP13:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[SHUFFLE3_I:%.*]] = shufflevector <8 x i64> [[TMP12]], <8 x i64> [[TMP13]], <8 x i32> <i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   store <8 x i64> [[SHUFFLE2_I]], <8 x i64>* [[__A_ADDR_I12_I]], align 64
// CHECK:   store <8 x i64> [[SHUFFLE3_I]], <8 x i64>* [[__B_ADDR_I13_I]], align 64
// CHECK:   [[TMP14:%.*]] = load <8 x i64>, <8 x i64>* [[__A_ADDR_I12_I]], align 64
// CHECK:   [[TMP15:%.*]] = load <8 x i64>, <8 x i64>* [[__B_ADDR_I13_I]], align 64
// CHECK:   [[TMP17:%.*]] = icmp sgt <8 x i64> [[TMP14]], [[TMP15]]
// CHECK:   [[TMP18:%.*]] = select <8 x i1> [[TMP17]], <8 x i64> [[TMP14]], <8 x i64> [[TMP15]]
// CHECK:   store <8 x i64> [[TMP18]], <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP19:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP20:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[SHUFFLE5_I:%.*]] = shufflevector <8 x i64> [[TMP19]], <8 x i64> [[TMP20]], <8 x i32> <i32 0, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP21:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP22:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[SHUFFLE6_I:%.*]] = shufflevector <8 x i64> [[TMP21]], <8 x i64> [[TMP22]], <8 x i32> <i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   store <8 x i64> [[SHUFFLE5_I]], <8 x i64>* [[__A_ADDR_I9_I]], align 64
// CHECK:   store <8 x i64> [[SHUFFLE6_I]], <8 x i64>* [[__B_ADDR_I10_I]], align 64
// CHECK:   [[TMP23:%.*]] = load <8 x i64>, <8 x i64>* [[__A_ADDR_I9_I]], align 64
// CHECK:   [[TMP24:%.*]] = load <8 x i64>, <8 x i64>* [[__B_ADDR_I10_I]], align 64
// CHECK:   [[TMP26:%.*]] = icmp sgt <8 x i64> [[TMP23]], [[TMP24]]
// CHECK:   [[TMP27:%.*]] = select <8 x i1> [[TMP26]], <8 x i64> [[TMP23]], <8 x i64> [[TMP24]]
// CHECK:   store <8 x i64> [[TMP27]], <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP28:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[VECEXT_I:%.*]] = extractelement <8 x i64> [[TMP28]], i32 0
// CHECK:   ret i64 [[VECEXT_I]]
long long test_mm512_reduce_max_epi64(__m512i __W){
  return _mm512_reduce_max_epi64(__W);
}

// CHECK-LABEL: define i64 @test_mm512_reduce_max_epu64(<8 x i64> %__W) #0 {
// CHECK:   [[__A_ADDR_I12_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__B_ADDR_I13_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__A_ADDR_I9_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__B_ADDR_I10_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__A_ADDR_I_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__B_ADDR_I_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__V_ADDR_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__W_ADDR:%.*]] = alloca <8 x i64>, align 64
// CHECK:   store <8 x i64> %__W, <8 x i64>* [[__W_ADDR]], align 64
// CHECK:   [[TMP0:%.*]] = load <8 x i64>, <8 x i64>* [[__W_ADDR]], align 64
// CHECK:   store <8 x i64> [[TMP0]], <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP1:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP2:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i64> [[TMP1]], <8 x i64> [[TMP2]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP3:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP4:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[SHUFFLE1_I:%.*]] = shufflevector <8 x i64> [[TMP3]], <8 x i64> [[TMP4]], <8 x i32> <i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   store <8 x i64> [[SHUFFLE_I]], <8 x i64>* [[__A_ADDR_I_I]], align 64
// CHECK:   store <8 x i64> [[SHUFFLE1_I]], <8 x i64>* [[__B_ADDR_I_I]], align 64
// CHECK:   [[TMP5:%.*]] = load <8 x i64>, <8 x i64>* [[__A_ADDR_I_I]], align 64
// CHECK:   [[TMP6:%.*]] = load <8 x i64>, <8 x i64>* [[__B_ADDR_I_I]], align 64
// CHECK:   [[TMP8:%.*]] = icmp ugt <8 x i64> [[TMP5]], [[TMP6]]
// CHECK:   [[TMP9:%.*]] = select <8 x i1> [[TMP8]], <8 x i64> [[TMP5]], <8 x i64> [[TMP6]]
// CHECK:   store <8 x i64> [[TMP9]], <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP10:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP11:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[SHUFFLE2_I:%.*]] = shufflevector <8 x i64> [[TMP10]], <8 x i64> [[TMP11]], <8 x i32> <i32 0, i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP12:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP13:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[SHUFFLE3_I:%.*]] = shufflevector <8 x i64> [[TMP12]], <8 x i64> [[TMP13]], <8 x i32> <i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   store <8 x i64> [[SHUFFLE2_I]], <8 x i64>* [[__A_ADDR_I12_I]], align 64
// CHECK:   store <8 x i64> [[SHUFFLE3_I]], <8 x i64>* [[__B_ADDR_I13_I]], align 64
// CHECK:   [[TMP14:%.*]] = load <8 x i64>, <8 x i64>* [[__A_ADDR_I12_I]], align 64
// CHECK:   [[TMP15:%.*]] = load <8 x i64>, <8 x i64>* [[__B_ADDR_I13_I]], align 64
// CHECK:   [[TMP17:%.*]] = icmp ugt <8 x i64> [[TMP14]], [[TMP15]]
// CHECK:   [[TMP18:%.*]] = select <8 x i1> [[TMP17]], <8 x i64> [[TMP14]], <8 x i64> [[TMP15]]
// CHECK:   store <8 x i64> [[TMP18]], <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP19:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP20:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[SHUFFLE5_I:%.*]] = shufflevector <8 x i64> [[TMP19]], <8 x i64> [[TMP20]], <8 x i32> <i32 0, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP21:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP22:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[SHUFFLE6_I:%.*]] = shufflevector <8 x i64> [[TMP21]], <8 x i64> [[TMP22]], <8 x i32> <i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   store <8 x i64> [[SHUFFLE5_I]], <8 x i64>* [[__A_ADDR_I9_I]], align 64
// CHECK:   store <8 x i64> [[SHUFFLE6_I]], <8 x i64>* [[__B_ADDR_I10_I]], align 64
// CHECK:   [[TMP23:%.*]] = load <8 x i64>, <8 x i64>* [[__A_ADDR_I9_I]], align 64
// CHECK:   [[TMP24:%.*]] = load <8 x i64>, <8 x i64>* [[__B_ADDR_I10_I]], align 64
// CHECK:   [[TMP26:%.*]] = icmp ugt <8 x i64> [[TMP23]], [[TMP24]]
// CHECK:   [[TMP27:%.*]] = select <8 x i1> [[TMP26]], <8 x i64> [[TMP23]], <8 x i64> [[TMP24]]
// CHECK:   store <8 x i64> [[TMP27]], <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP28:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[VECEXT_I:%.*]] = extractelement <8 x i64> [[TMP28]], i32 0
// CHECK:   ret i64 [[VECEXT_I]]
unsigned long long test_mm512_reduce_max_epu64(__m512i __W){
  return _mm512_reduce_max_epu64(__W); 
}

// CHECK-LABEL: define double @test_mm512_reduce_max_pd(<8 x double> %__W) #0 {
// CHECK:   [[_COMPOUNDLITERAL_I_I11_I:%.*]] = alloca <8 x double>, align 64
// CHECK:   [[__A_ADDR_I12_I:%.*]] = alloca <8 x double>, align 64
// CHECK:   [[__B_ADDR_I13_I:%.*]] = alloca <8 x double>, align 64
// CHECK:   [[_COMPOUNDLITERAL_I_I8_I:%.*]] = alloca <8 x double>, align 64
// CHECK:   [[__A_ADDR_I9_I:%.*]] = alloca <8 x double>, align 64
// CHECK:   [[__B_ADDR_I10_I:%.*]] = alloca <8 x double>, align 64
// CHECK:   [[_COMPOUNDLITERAL_I_I_I:%.*]] = alloca <8 x double>, align 64
// CHECK:   [[__A_ADDR_I_I:%.*]] = alloca <8 x double>, align 64
// CHECK:   [[__B_ADDR_I_I:%.*]] = alloca <8 x double>, align 64
// CHECK:   [[__V_ADDR_I:%.*]] = alloca <8 x double>, align 64
// CHECK:   [[__W_ADDR:%.*]] = alloca <8 x double>, align 64
// CHECK:   store <8 x double> %__W, <8 x double>* [[__W_ADDR]], align 64
// CHECK:   [[TMP0:%.*]] = load <8 x double>, <8 x double>* [[__W_ADDR]], align 64
// CHECK:   store <8 x double> [[TMP0]], <8 x double>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP1:%.*]] = load <8 x double>, <8 x double>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP2:%.*]] = load <8 x double>, <8 x double>* [[__V_ADDR_I]], align 64
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x double> [[TMP1]], <8 x double> [[TMP2]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP3:%.*]] = load <8 x double>, <8 x double>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP4:%.*]] = load <8 x double>, <8 x double>* [[__V_ADDR_I]], align 64
// CHECK:   [[SHUFFLE1_I:%.*]] = shufflevector <8 x double> [[TMP3]], <8 x double> [[TMP4]], <8 x i32> <i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   store <8 x double> [[SHUFFLE_I]], <8 x double>* [[__A_ADDR_I_I]], align 64
// CHECK:   store <8 x double> [[SHUFFLE1_I]], <8 x double>* [[__B_ADDR_I_I]], align 64
// CHECK:   [[TMP5:%.*]] = load <8 x double>, <8 x double>* [[__A_ADDR_I_I]], align 64
// CHECK:   [[TMP6:%.*]] = load <8 x double>, <8 x double>* [[__B_ADDR_I_I]], align 64
// CHECK:   store <8 x double> zeroinitializer, <8 x double>* [[_COMPOUNDLITERAL_I_I_I]], align 64
// CHECK:   [[TMP7:%.*]] = load <8 x double>, <8 x double>* [[_COMPOUNDLITERAL_I_I_I]], align 64
// CHECK:   [[TMP8:%.*]] = call <8 x double> @llvm.x86.avx512.mask.max.pd.512(<8 x double> [[TMP5]], <8 x double> [[TMP6]], <8 x double> [[TMP7]], i8 -1, i32 4) #2
// CHECK:   store <8 x double> [[TMP8]], <8 x double>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP9:%.*]] = load <8 x double>, <8 x double>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP10:%.*]] = load <8 x double>, <8 x double>* [[__V_ADDR_I]], align 64
// CHECK:   [[SHUFFLE2_I:%.*]] = shufflevector <8 x double> [[TMP9]], <8 x double> [[TMP10]], <8 x i32> <i32 0, i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP11:%.*]] = load <8 x double>, <8 x double>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP12:%.*]] = load <8 x double>, <8 x double>* [[__V_ADDR_I]], align 64
// CHECK:   [[SHUFFLE3_I:%.*]] = shufflevector <8 x double> [[TMP11]], <8 x double> [[TMP12]], <8 x i32> <i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   store <8 x double> [[SHUFFLE2_I]], <8 x double>* [[__A_ADDR_I12_I]], align 64
// CHECK:   store <8 x double> [[SHUFFLE3_I]], <8 x double>* [[__B_ADDR_I13_I]], align 64
// CHECK:   [[TMP13:%.*]] = load <8 x double>, <8 x double>* [[__A_ADDR_I12_I]], align 64
// CHECK:   [[TMP14:%.*]] = load <8 x double>, <8 x double>* [[__B_ADDR_I13_I]], align 64
// CHECK:   store <8 x double> zeroinitializer, <8 x double>* [[_COMPOUNDLITERAL_I_I11_I]], align 64
// CHECK:   [[TMP15:%.*]] = load <8 x double>, <8 x double>* [[_COMPOUNDLITERAL_I_I11_I]], align 64
// CHECK:   [[TMP16:%.*]] = call <8 x double> @llvm.x86.avx512.mask.max.pd.512(<8 x double> [[TMP13]], <8 x double> [[TMP14]], <8 x double> [[TMP15]], i8 -1, i32 4) #2
// CHECK:   store <8 x double> [[TMP16]], <8 x double>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP17:%.*]] = load <8 x double>, <8 x double>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP18:%.*]] = load <8 x double>, <8 x double>* [[__V_ADDR_I]], align 64
// CHECK:   [[SHUFFLE5_I:%.*]] = shufflevector <8 x double> [[TMP17]], <8 x double> [[TMP18]], <8 x i32> <i32 0, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP19:%.*]] = load <8 x double>, <8 x double>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP20:%.*]] = load <8 x double>, <8 x double>* [[__V_ADDR_I]], align 64
// CHECK:   [[SHUFFLE6_I:%.*]] = shufflevector <8 x double> [[TMP19]], <8 x double> [[TMP20]], <8 x i32> <i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   store <8 x double> [[SHUFFLE5_I]], <8 x double>* [[__A_ADDR_I9_I]], align 64
// CHECK:   store <8 x double> [[SHUFFLE6_I]], <8 x double>* [[__B_ADDR_I10_I]], align 64
// CHECK:   [[TMP21:%.*]] = load <8 x double>, <8 x double>* [[__A_ADDR_I9_I]], align 64
// CHECK:   [[TMP22:%.*]] = load <8 x double>, <8 x double>* [[__B_ADDR_I10_I]], align 64
// CHECK:   store <8 x double> zeroinitializer, <8 x double>* [[_COMPOUNDLITERAL_I_I8_I]], align 64
// CHECK:   [[TMP23:%.*]] = load <8 x double>, <8 x double>* [[_COMPOUNDLITERAL_I_I8_I]], align 64
// CHECK:   [[TMP24:%.*]] = call <8 x double> @llvm.x86.avx512.mask.max.pd.512(<8 x double> [[TMP21]], <8 x double> [[TMP22]], <8 x double> [[TMP23]], i8 -1, i32 4) #2
// CHECK:   store <8 x double> [[TMP24]], <8 x double>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP25:%.*]] = load <8 x double>, <8 x double>* [[__V_ADDR_I]], align 64
// CHECK:   [[VECEXT_I:%.*]] = extractelement <8 x double> [[TMP25]], i32 0
// CHECK:   ret double [[VECEXT_I]]
double test_mm512_reduce_max_pd(__m512d __W){
  return _mm512_reduce_max_pd(__W); 
}

// CHECK-LABEL: define i64 @test_mm512_reduce_min_epi64(<8 x i64> %__W) #0 {
// CHECK:   [[__A_ADDR_I12_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__B_ADDR_I13_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__A_ADDR_I9_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__B_ADDR_I10_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__A_ADDR_I_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__B_ADDR_I_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__V_ADDR_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__W_ADDR:%.*]] = alloca <8 x i64>, align 64
// CHECK:   store <8 x i64> %__W, <8 x i64>* [[__W_ADDR]], align 64
// CHECK:   [[TMP0:%.*]] = load <8 x i64>, <8 x i64>* [[__W_ADDR]], align 64
// CHECK:   store <8 x i64> [[TMP0]], <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP1:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP2:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i64> [[TMP1]], <8 x i64> [[TMP2]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP3:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP4:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[SHUFFLE1_I:%.*]] = shufflevector <8 x i64> [[TMP3]], <8 x i64> [[TMP4]], <8 x i32> <i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   store <8 x i64> [[SHUFFLE_I]], <8 x i64>* [[__A_ADDR_I_I]], align 64
// CHECK:   store <8 x i64> [[SHUFFLE1_I]], <8 x i64>* [[__B_ADDR_I_I]], align 64
// CHECK:   [[TMP5:%.*]] = load <8 x i64>, <8 x i64>* [[__A_ADDR_I_I]], align 64
// CHECK:   [[TMP6:%.*]] = load <8 x i64>, <8 x i64>* [[__B_ADDR_I_I]], align 64
// CHECK:   [[TMP8:%.*]] = icmp slt <8 x i64> [[TMP5]], [[TMP6]]
// CHECK:   [[TMP9:%.*]] = select <8 x i1> [[TMP8]], <8 x i64> [[TMP5]], <8 x i64> [[TMP6]]
// CHECK:   store <8 x i64> [[TMP9]], <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP10:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP11:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[SHUFFLE2_I:%.*]] = shufflevector <8 x i64> [[TMP10]], <8 x i64> [[TMP11]], <8 x i32> <i32 0, i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP12:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP13:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[SHUFFLE3_I:%.*]] = shufflevector <8 x i64> [[TMP12]], <8 x i64> [[TMP13]], <8 x i32> <i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   store <8 x i64> [[SHUFFLE2_I]], <8 x i64>* [[__A_ADDR_I12_I]], align 64
// CHECK:   store <8 x i64> [[SHUFFLE3_I]], <8 x i64>* [[__B_ADDR_I13_I]], align 64
// CHECK:   [[TMP14:%.*]] = load <8 x i64>, <8 x i64>* [[__A_ADDR_I12_I]], align 64
// CHECK:   [[TMP15:%.*]] = load <8 x i64>, <8 x i64>* [[__B_ADDR_I13_I]], align 64
// CHECK:   [[TMP17:%.*]] = icmp slt <8 x i64> [[TMP14]], [[TMP15]]
// CHECK:   [[TMP18:%.*]] = select <8 x i1> [[TMP17]], <8 x i64> [[TMP14]], <8 x i64> [[TMP15]]
// CHECK:   store <8 x i64> [[TMP18]], <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP19:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP20:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[SHUFFLE5_I:%.*]] = shufflevector <8 x i64> [[TMP19]], <8 x i64> [[TMP20]], <8 x i32> <i32 0, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP21:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP22:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[SHUFFLE6_I:%.*]] = shufflevector <8 x i64> [[TMP21]], <8 x i64> [[TMP22]], <8 x i32> <i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   store <8 x i64> [[SHUFFLE5_I]], <8 x i64>* [[__A_ADDR_I9_I]], align 64
// CHECK:   store <8 x i64> [[SHUFFLE6_I]], <8 x i64>* [[__B_ADDR_I10_I]], align 64
// CHECK:   [[TMP23:%.*]] = load <8 x i64>, <8 x i64>* [[__A_ADDR_I9_I]], align 64
// CHECK:   [[TMP24:%.*]] = load <8 x i64>, <8 x i64>* [[__B_ADDR_I10_I]], align 64
// CHECK:   [[TMP26:%.*]] = icmp slt <8 x i64> [[TMP23]], [[TMP24]]
// CHECK:   [[TMP27:%.*]] = select <8 x i1> [[TMP26]], <8 x i64> [[TMP23]], <8 x i64> [[TMP24]]
// CHECK:   store <8 x i64> [[TMP27]], <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP28:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[VECEXT_I:%.*]] = extractelement <8 x i64> [[TMP28]], i32 0
// CHECK:   ret i64 [[VECEXT_I]]
long long test_mm512_reduce_min_epi64(__m512i __W){
  return _mm512_reduce_min_epi64(__W);
}

// CHECK-LABEL: define i64 @test_mm512_reduce_min_epu64(<8 x i64> %__W) #0 {
// CHECK:   [[__A_ADDR_I12_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__B_ADDR_I13_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__A_ADDR_I9_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__B_ADDR_I10_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__A_ADDR_I_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__B_ADDR_I_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__V_ADDR_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__W_ADDR:%.*]] = alloca <8 x i64>, align 64
// CHECK:   store <8 x i64> %__W, <8 x i64>* [[__W_ADDR]], align 64
// CHECK:   [[TMP0:%.*]] = load <8 x i64>, <8 x i64>* [[__W_ADDR]], align 64
// CHECK:   store <8 x i64> [[TMP0]], <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP1:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP2:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i64> [[TMP1]], <8 x i64> [[TMP2]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP3:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP4:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[SHUFFLE1_I:%.*]] = shufflevector <8 x i64> [[TMP3]], <8 x i64> [[TMP4]], <8 x i32> <i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   store <8 x i64> [[SHUFFLE_I]], <8 x i64>* [[__A_ADDR_I_I]], align 64
// CHECK:   store <8 x i64> [[SHUFFLE1_I]], <8 x i64>* [[__B_ADDR_I_I]], align 64
// CHECK:   [[TMP5:%.*]] = load <8 x i64>, <8 x i64>* [[__A_ADDR_I_I]], align 64
// CHECK:   [[TMP6:%.*]] = load <8 x i64>, <8 x i64>* [[__B_ADDR_I_I]], align 64
// CHECK:   [[TMP8:%.*]] = icmp ult <8 x i64> [[TMP5]], [[TMP6]]
// CHECK:   [[TMP9:%.*]] = select <8 x i1> [[TMP8]], <8 x i64> [[TMP5]], <8 x i64> [[TMP6]]
// CHECK:   store <8 x i64> [[TMP9]], <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP10:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP11:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[SHUFFLE2_I:%.*]] = shufflevector <8 x i64> [[TMP10]], <8 x i64> [[TMP11]], <8 x i32> <i32 0, i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP12:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP13:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[SHUFFLE3_I:%.*]] = shufflevector <8 x i64> [[TMP12]], <8 x i64> [[TMP13]], <8 x i32> <i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   store <8 x i64> [[SHUFFLE2_I]], <8 x i64>* [[__A_ADDR_I12_I]], align 64
// CHECK:   store <8 x i64> [[SHUFFLE3_I]], <8 x i64>* [[__B_ADDR_I13_I]], align 64
// CHECK:   [[TMP14:%.*]] = load <8 x i64>, <8 x i64>* [[__A_ADDR_I12_I]], align 64
// CHECK:   [[TMP15:%.*]] = load <8 x i64>, <8 x i64>* [[__B_ADDR_I13_I]], align 64
// CHECK:   [[TMP17:%.*]] = icmp ult <8 x i64> [[TMP14]], [[TMP15]]
// CHECK:   [[TMP18:%.*]] = select <8 x i1> [[TMP17]], <8 x i64> [[TMP14]], <8 x i64> [[TMP15]]
// CHECK:   store <8 x i64> [[TMP18]], <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP19:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP20:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[SHUFFLE5_I:%.*]] = shufflevector <8 x i64> [[TMP19]], <8 x i64> [[TMP20]], <8 x i32> <i32 0, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP21:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP22:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[SHUFFLE6_I:%.*]] = shufflevector <8 x i64> [[TMP21]], <8 x i64> [[TMP22]], <8 x i32> <i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   store <8 x i64> [[SHUFFLE5_I]], <8 x i64>* [[__A_ADDR_I9_I]], align 64
// CHECK:   store <8 x i64> [[SHUFFLE6_I]], <8 x i64>* [[__B_ADDR_I10_I]], align 64
// CHECK:   [[TMP23:%.*]] = load <8 x i64>, <8 x i64>* [[__A_ADDR_I9_I]], align 64
// CHECK:   [[TMP24:%.*]] = load <8 x i64>, <8 x i64>* [[__B_ADDR_I10_I]], align 64
// CHECK:   [[TMP26:%.*]] = icmp ult <8 x i64> [[TMP23]], [[TMP24]]
// CHECK:   [[TMP27:%.*]] = select <8 x i1> [[TMP26]], <8 x i64> [[TMP23]], <8 x i64> [[TMP24]]
// CHECK:   store <8 x i64> [[TMP27]], <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP28:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[VECEXT_I:%.*]] = extractelement <8 x i64> [[TMP28]], i32 0
// CHECK:   ret i64 [[VECEXT_I]]
unsigned long long test_mm512_reduce_min_epu64(__m512i __W){
  return _mm512_reduce_min_epu64(__W);
}

// CHECK-LABEL: define double @test_mm512_reduce_min_pd(<8 x double> %__W) #0 {
// CHECK:   [[_COMPOUNDLITERAL_I_I11_I:%.*]] = alloca <8 x double>, align 64
// CHECK:   [[__A_ADDR_I12_I:%.*]] = alloca <8 x double>, align 64
// CHECK:   [[__B_ADDR_I13_I:%.*]] = alloca <8 x double>, align 64
// CHECK:   [[_COMPOUNDLITERAL_I_I8_I:%.*]] = alloca <8 x double>, align 64
// CHECK:   [[__A_ADDR_I9_I:%.*]] = alloca <8 x double>, align 64
// CHECK:   [[__B_ADDR_I10_I:%.*]] = alloca <8 x double>, align 64
// CHECK:   [[_COMPOUNDLITERAL_I_I_I:%.*]] = alloca <8 x double>, align 64
// CHECK:   [[__A_ADDR_I_I:%.*]] = alloca <8 x double>, align 64
// CHECK:   [[__B_ADDR_I_I:%.*]] = alloca <8 x double>, align 64
// CHECK:   [[__V_ADDR_I:%.*]] = alloca <8 x double>, align 64
// CHECK:   [[__W_ADDR:%.*]] = alloca <8 x double>, align 64
// CHECK:   store <8 x double> %__W, <8 x double>* [[__W_ADDR]], align 64
// CHECK:   [[TMP0:%.*]] = load <8 x double>, <8 x double>* [[__W_ADDR]], align 64
// CHECK:   store <8 x double> [[TMP0]], <8 x double>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP1:%.*]] = load <8 x double>, <8 x double>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP2:%.*]] = load <8 x double>, <8 x double>* [[__V_ADDR_I]], align 64
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x double> [[TMP1]], <8 x double> [[TMP2]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP3:%.*]] = load <8 x double>, <8 x double>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP4:%.*]] = load <8 x double>, <8 x double>* [[__V_ADDR_I]], align 64
// CHECK:   [[SHUFFLE1_I:%.*]] = shufflevector <8 x double> [[TMP3]], <8 x double> [[TMP4]], <8 x i32> <i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   store <8 x double> [[SHUFFLE_I]], <8 x double>* [[__A_ADDR_I_I]], align 64
// CHECK:   store <8 x double> [[SHUFFLE1_I]], <8 x double>* [[__B_ADDR_I_I]], align 64
// CHECK:   [[TMP5:%.*]] = load <8 x double>, <8 x double>* [[__A_ADDR_I_I]], align 64
// CHECK:   [[TMP6:%.*]] = load <8 x double>, <8 x double>* [[__B_ADDR_I_I]], align 64
// CHECK:   store <8 x double> zeroinitializer, <8 x double>* [[_COMPOUNDLITERAL_I_I_I]], align 64
// CHECK:   [[TMP7:%.*]] = load <8 x double>, <8 x double>* [[_COMPOUNDLITERAL_I_I_I]], align 64
// CHECK:   [[TMP8:%.*]] = call <8 x double> @llvm.x86.avx512.mask.min.pd.512(<8 x double> [[TMP5]], <8 x double> [[TMP6]], <8 x double> [[TMP7]], i8 -1, i32 4) #2
// CHECK:   store <8 x double> [[TMP8]], <8 x double>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP9:%.*]] = load <8 x double>, <8 x double>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP10:%.*]] = load <8 x double>, <8 x double>* [[__V_ADDR_I]], align 64
// CHECK:   [[SHUFFLE2_I:%.*]] = shufflevector <8 x double> [[TMP9]], <8 x double> [[TMP10]], <8 x i32> <i32 0, i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP11:%.*]] = load <8 x double>, <8 x double>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP12:%.*]] = load <8 x double>, <8 x double>* [[__V_ADDR_I]], align 64
// CHECK:   [[SHUFFLE3_I:%.*]] = shufflevector <8 x double> [[TMP11]], <8 x double> [[TMP12]], <8 x i32> <i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   store <8 x double> [[SHUFFLE2_I]], <8 x double>* [[__A_ADDR_I12_I]], align 64
// CHECK:   store <8 x double> [[SHUFFLE3_I]], <8 x double>* [[__B_ADDR_I13_I]], align 64
// CHECK:   [[TMP13:%.*]] = load <8 x double>, <8 x double>* [[__A_ADDR_I12_I]], align 64
// CHECK:   [[TMP14:%.*]] = load <8 x double>, <8 x double>* [[__B_ADDR_I13_I]], align 64
// CHECK:   store <8 x double> zeroinitializer, <8 x double>* [[_COMPOUNDLITERAL_I_I11_I]], align 64
// CHECK:   [[TMP15:%.*]] = load <8 x double>, <8 x double>* [[_COMPOUNDLITERAL_I_I11_I]], align 64
// CHECK:   [[TMP16:%.*]] = call <8 x double> @llvm.x86.avx512.mask.min.pd.512(<8 x double> [[TMP13]], <8 x double> [[TMP14]], <8 x double> [[TMP15]], i8 -1, i32 4) #2
// CHECK:   store <8 x double> [[TMP16]], <8 x double>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP17:%.*]] = load <8 x double>, <8 x double>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP18:%.*]] = load <8 x double>, <8 x double>* [[__V_ADDR_I]], align 64
// CHECK:   [[SHUFFLE5_I:%.*]] = shufflevector <8 x double> [[TMP17]], <8 x double> [[TMP18]], <8 x i32> <i32 0, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP19:%.*]] = load <8 x double>, <8 x double>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP20:%.*]] = load <8 x double>, <8 x double>* [[__V_ADDR_I]], align 64
// CHECK:   [[SHUFFLE6_I:%.*]] = shufflevector <8 x double> [[TMP19]], <8 x double> [[TMP20]], <8 x i32> <i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   store <8 x double> [[SHUFFLE5_I]], <8 x double>* [[__A_ADDR_I9_I]], align 64
// CHECK:   store <8 x double> [[SHUFFLE6_I]], <8 x double>* [[__B_ADDR_I10_I]], align 64
// CHECK:   [[TMP21:%.*]] = load <8 x double>, <8 x double>* [[__A_ADDR_I9_I]], align 64
// CHECK:   [[TMP22:%.*]] = load <8 x double>, <8 x double>* [[__B_ADDR_I10_I]], align 64
// CHECK:   store <8 x double> zeroinitializer, <8 x double>* [[_COMPOUNDLITERAL_I_I8_I]], align 64
// CHECK:   [[TMP23:%.*]] = load <8 x double>, <8 x double>* [[_COMPOUNDLITERAL_I_I8_I]], align 64
// CHECK:   [[TMP24:%.*]] = call <8 x double> @llvm.x86.avx512.mask.min.pd.512(<8 x double> [[TMP21]], <8 x double> [[TMP22]], <8 x double> [[TMP23]], i8 -1, i32 4) #2
// CHECK:   store <8 x double> [[TMP24]], <8 x double>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP25:%.*]] = load <8 x double>, <8 x double>* [[__V_ADDR_I]], align 64
// CHECK:   [[VECEXT_I:%.*]] = extractelement <8 x double> [[TMP25]], i32 0
// CHECK:   ret double [[VECEXT_I]]
double test_mm512_reduce_min_pd(__m512d __W){
  return _mm512_reduce_min_pd(__W); 
}

// CHECK-LABEL: define i64 @test_mm512_mask_reduce_max_epi64(i8 zeroext %__M, <8 x i64> %__W) #0 {
// CHECK:   [[__A_ADDR_I13_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__B_ADDR_I14_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__A_ADDR_I10_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__B_ADDR_I11_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__A_ADDR_I_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__B_ADDR_I_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__D_ADDR_I_I:%.*]] = alloca i64, align 8
// CHECK:   [[_COMPOUNDLITERAL_I_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__M_ADDR_I:%.*]] = alloca i8, align 1
// CHECK:   [[__V_ADDR_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__M_ADDR:%.*]] = alloca i8, align 1
// CHECK:   [[__W_ADDR:%.*]] = alloca <8 x i64>, align 64
// CHECK:   store i8 %__M, i8* [[__M_ADDR]], align 1
// CHECK:   store <8 x i64> %__W, <8 x i64>* [[__W_ADDR]], align 64
// CHECK:   [[TMP0:%.*]] = load i8, i8* [[__M_ADDR]], align 1
// CHECK:   [[TMP1:%.*]] = load <8 x i64>, <8 x i64>* [[__W_ADDR]], align 64
// CHECK:   store i8 [[TMP0]], i8* [[__M_ADDR_I]], align 1
// CHECK:   store <8 x i64> [[TMP1]], <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP2:%.*]] = load i8, i8* [[__M_ADDR_I]], align 1
// CHECK:   [[TMP3:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   store i64 -9223372036854775808, i64* [[__D_ADDR_I_I]], align 8
// CHECK:   [[TMP4:%.*]] = load i64, i64* [[__D_ADDR_I_I]], align 8
// CHECK:   [[VECINIT_I_I:%.*]] = insertelement <8 x i64> undef, i64 [[TMP4]], i32 0
// CHECK:   [[TMP5:%.*]] = load i64, i64* [[__D_ADDR_I_I]], align 8
// CHECK:   [[VECINIT1_I_I:%.*]] = insertelement <8 x i64> [[VECINIT_I_I]], i64 [[TMP5]], i32 1
// CHECK:   [[TMP6:%.*]] = load i64, i64* [[__D_ADDR_I_I]], align 8
// CHECK:   [[VECINIT2_I_I:%.*]] = insertelement <8 x i64> [[VECINIT1_I_I]], i64 [[TMP6]], i32 2
// CHECK:   [[TMP7:%.*]] = load i64, i64* [[__D_ADDR_I_I]], align 8
// CHECK:   [[VECINIT3_I_I:%.*]] = insertelement <8 x i64> [[VECINIT2_I_I]], i64 [[TMP7]], i32 3
// CHECK:   [[TMP8:%.*]] = load i64, i64* [[__D_ADDR_I_I]], align 8
// CHECK:   [[VECINIT4_I_I:%.*]] = insertelement <8 x i64> [[VECINIT3_I_I]], i64 [[TMP8]], i32 4
// CHECK:   [[TMP9:%.*]] = load i64, i64* [[__D_ADDR_I_I]], align 8
// CHECK:   [[VECINIT5_I_I:%.*]] = insertelement <8 x i64> [[VECINIT4_I_I]], i64 [[TMP9]], i32 5
// CHECK:   [[TMP10:%.*]] = load i64, i64* [[__D_ADDR_I_I]], align 8
// CHECK:   [[VECINIT6_I_I:%.*]] = insertelement <8 x i64> [[VECINIT5_I_I]], i64 [[TMP10]], i32 6
// CHECK:   [[TMP11:%.*]] = load i64, i64* [[__D_ADDR_I_I]], align 8
// CHECK:   [[VECINIT7_I_I:%.*]] = insertelement <8 x i64> [[VECINIT6_I_I]], i64 [[TMP11]], i32 7
// CHECK:   store <8 x i64> [[VECINIT7_I_I]], <8 x i64>* [[_COMPOUNDLITERAL_I_I]], align 64
// CHECK:   [[TMP12:%.*]] = load <8 x i64>, <8 x i64>* [[_COMPOUNDLITERAL_I_I]], align 64
// CHECK:   [[TMP13:%.*]] = bitcast i8 [[TMP2]] to <8 x i1>
// CHECK:   [[TMP14:%.*]] = select <8 x i1> [[TMP13]], <8 x i64> [[TMP3]], <8 x i64> [[TMP12]]
// CHECK:   store <8 x i64> [[TMP14]], <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP15:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP16:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i64> [[TMP15]], <8 x i64> [[TMP16]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP17:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP18:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[SHUFFLE1_I:%.*]] = shufflevector <8 x i64> [[TMP17]], <8 x i64> [[TMP18]], <8 x i32> <i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   store <8 x i64> [[SHUFFLE_I]], <8 x i64>* [[__A_ADDR_I13_I]], align 64
// CHECK:   store <8 x i64> [[SHUFFLE1_I]], <8 x i64>* [[__B_ADDR_I14_I]], align 64
// CHECK:   [[TMP19:%.*]] = load <8 x i64>, <8 x i64>* [[__A_ADDR_I13_I]], align 64
// CHECK:   [[TMP20:%.*]] = load <8 x i64>, <8 x i64>* [[__B_ADDR_I14_I]], align 64
// CHECK:   [[TMP22:%.*]] = icmp sgt <8 x i64> [[TMP19]], [[TMP20]]
// CHECK:   [[TMP23:%.*]] = select <8 x i1> [[TMP22]], <8 x i64> [[TMP19]], <8 x i64> [[TMP20]]
// CHECK:   store <8 x i64> [[TMP23]], <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP24:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP25:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[SHUFFLE3_I:%.*]] = shufflevector <8 x i64> [[TMP24]], <8 x i64> [[TMP25]], <8 x i32> <i32 0, i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP26:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP27:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[SHUFFLE4_I:%.*]] = shufflevector <8 x i64> [[TMP26]], <8 x i64> [[TMP27]], <8 x i32> <i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   store <8 x i64> [[SHUFFLE3_I]], <8 x i64>* [[__A_ADDR_I10_I]], align 64
// CHECK:   store <8 x i64> [[SHUFFLE4_I]], <8 x i64>* [[__B_ADDR_I11_I]], align 64
// CHECK:   [[TMP28:%.*]] = load <8 x i64>, <8 x i64>* [[__A_ADDR_I10_I]], align 64
// CHECK:   [[TMP29:%.*]] = load <8 x i64>, <8 x i64>* [[__B_ADDR_I11_I]], align 64
// CHECK:   [[TMP31:%.*]] = icmp sgt <8 x i64> [[TMP28]], [[TMP29]]
// CHECK:   [[TMP32:%.*]] = select <8 x i1> [[TMP31]], <8 x i64> [[TMP28]], <8 x i64> [[TMP29]]
// CHECK:   store <8 x i64> [[TMP32]], <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP33:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP34:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[SHUFFLE6_I:%.*]] = shufflevector <8 x i64> [[TMP33]], <8 x i64> [[TMP34]], <8 x i32> <i32 0, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP35:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP36:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[SHUFFLE7_I:%.*]] = shufflevector <8 x i64> [[TMP35]], <8 x i64> [[TMP36]], <8 x i32> <i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   store <8 x i64> [[SHUFFLE6_I]], <8 x i64>* [[__A_ADDR_I_I]], align 64
// CHECK:   store <8 x i64> [[SHUFFLE7_I]], <8 x i64>* [[__B_ADDR_I_I]], align 64
// CHECK:   [[TMP37:%.*]] = load <8 x i64>, <8 x i64>* [[__A_ADDR_I_I]], align 64
// CHECK:   [[TMP38:%.*]] = load <8 x i64>, <8 x i64>* [[__B_ADDR_I_I]], align 64
// CHECK:   [[TMP40:%.*]] = icmp sgt <8 x i64> [[TMP37]], [[TMP38]]
// CHECK:   [[TMP41:%.*]] = select <8 x i1> [[TMP40]], <8 x i64> [[TMP37]], <8 x i64> [[TMP38]]
// CHECK:   store <8 x i64> [[TMP41]], <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP42:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[VECEXT_I:%.*]] = extractelement <8 x i64> [[TMP42]], i32 0
// CHECK:   ret i64 [[VECEXT_I]]
long long test_mm512_mask_reduce_max_epi64(__mmask8 __M, __m512i __W){
  return _mm512_mask_reduce_max_epi64(__M, __W); 
}

// CHECK-LABEL: define i64 @test_mm512_mask_reduce_max_epu64(i8 zeroext %__M, <8 x i64> %__W) #0 {
// CHECK:   [[__A_ADDR_I13_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__B_ADDR_I14_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__A_ADDR_I10_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__B_ADDR_I11_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__A_ADDR_I_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__B_ADDR_I_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__D_ADDR_I_I:%.*]] = alloca i64, align 8
// CHECK:   [[_COMPOUNDLITERAL_I_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__M_ADDR_I:%.*]] = alloca i8, align 1
// CHECK:   [[__V_ADDR_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__M_ADDR:%.*]] = alloca i8, align 1
// CHECK:   [[__W_ADDR:%.*]] = alloca <8 x i64>, align 64
// CHECK:   store i8 %__M, i8* [[__M_ADDR]], align 1
// CHECK:   store <8 x i64> %__W, <8 x i64>* [[__W_ADDR]], align 64
// CHECK:   [[TMP0:%.*]] = load i8, i8* [[__M_ADDR]], align 1
// CHECK:   [[TMP1:%.*]] = load <8 x i64>, <8 x i64>* [[__W_ADDR]], align 64
// CHECK:   store i8 [[TMP0]], i8* [[__M_ADDR_I]], align 1
// CHECK:   store <8 x i64> [[TMP1]], <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP2:%.*]] = load i8, i8* [[__M_ADDR_I]], align 1
// CHECK:   [[TMP3:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   store i64 0, i64* [[__D_ADDR_I_I]], align 8
// CHECK:   [[TMP4:%.*]] = load i64, i64* [[__D_ADDR_I_I]], align 8
// CHECK:   [[VECINIT_I_I:%.*]] = insertelement <8 x i64> undef, i64 [[TMP4]], i32 0
// CHECK:   [[TMP5:%.*]] = load i64, i64* [[__D_ADDR_I_I]], align 8
// CHECK:   [[VECINIT1_I_I:%.*]] = insertelement <8 x i64> [[VECINIT_I_I]], i64 [[TMP5]], i32 1
// CHECK:   [[TMP6:%.*]] = load i64, i64* [[__D_ADDR_I_I]], align 8
// CHECK:   [[VECINIT2_I_I:%.*]] = insertelement <8 x i64> [[VECINIT1_I_I]], i64 [[TMP6]], i32 2
// CHECK:   [[TMP7:%.*]] = load i64, i64* [[__D_ADDR_I_I]], align 8
// CHECK:   [[VECINIT3_I_I:%.*]] = insertelement <8 x i64> [[VECINIT2_I_I]], i64 [[TMP7]], i32 3
// CHECK:   [[TMP8:%.*]] = load i64, i64* [[__D_ADDR_I_I]], align 8
// CHECK:   [[VECINIT4_I_I:%.*]] = insertelement <8 x i64> [[VECINIT3_I_I]], i64 [[TMP8]], i32 4
// CHECK:   [[TMP9:%.*]] = load i64, i64* [[__D_ADDR_I_I]], align 8
// CHECK:   [[VECINIT5_I_I:%.*]] = insertelement <8 x i64> [[VECINIT4_I_I]], i64 [[TMP9]], i32 5
// CHECK:   [[TMP10:%.*]] = load i64, i64* [[__D_ADDR_I_I]], align 8
// CHECK:   [[VECINIT6_I_I:%.*]] = insertelement <8 x i64> [[VECINIT5_I_I]], i64 [[TMP10]], i32 6
// CHECK:   [[TMP11:%.*]] = load i64, i64* [[__D_ADDR_I_I]], align 8
// CHECK:   [[VECINIT7_I_I:%.*]] = insertelement <8 x i64> [[VECINIT6_I_I]], i64 [[TMP11]], i32 7
// CHECK:   store <8 x i64> [[VECINIT7_I_I]], <8 x i64>* [[_COMPOUNDLITERAL_I_I]], align 64
// CHECK:   [[TMP12:%.*]] = load <8 x i64>, <8 x i64>* [[_COMPOUNDLITERAL_I_I]], align 64
// CHECK:   [[TMP13:%.*]] = bitcast i8 [[TMP2]] to <8 x i1>
// CHECK:   [[TMP14:%.*]] = select <8 x i1> [[TMP13]], <8 x i64> [[TMP3]], <8 x i64> [[TMP12]]
// CHECK:   store <8 x i64> [[TMP14]], <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP15:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP16:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i64> [[TMP15]], <8 x i64> [[TMP16]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP17:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP18:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[SHUFFLE1_I:%.*]] = shufflevector <8 x i64> [[TMP17]], <8 x i64> [[TMP18]], <8 x i32> <i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   store <8 x i64> [[SHUFFLE_I]], <8 x i64>* [[__A_ADDR_I13_I]], align 64
// CHECK:   store <8 x i64> [[SHUFFLE1_I]], <8 x i64>* [[__B_ADDR_I14_I]], align 64
// CHECK:   [[TMP19:%.*]] = load <8 x i64>, <8 x i64>* [[__A_ADDR_I13_I]], align 64
// CHECK:   [[TMP20:%.*]] = load <8 x i64>, <8 x i64>* [[__B_ADDR_I14_I]], align 64
// CHECK:   [[TMP22:%.*]] = icmp ugt <8 x i64> [[TMP19]], [[TMP20]]
// CHECK:   [[TMP23:%.*]] = select <8 x i1> [[TMP22]], <8 x i64> [[TMP19]], <8 x i64> [[TMP20]]
// CHECK:   store <8 x i64> [[TMP23]], <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP24:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP25:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[SHUFFLE3_I:%.*]] = shufflevector <8 x i64> [[TMP24]], <8 x i64> [[TMP25]], <8 x i32> <i32 0, i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP26:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP27:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[SHUFFLE4_I:%.*]] = shufflevector <8 x i64> [[TMP26]], <8 x i64> [[TMP27]], <8 x i32> <i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   store <8 x i64> [[SHUFFLE3_I]], <8 x i64>* [[__A_ADDR_I10_I]], align 64
// CHECK:   store <8 x i64> [[SHUFFLE4_I]], <8 x i64>* [[__B_ADDR_I11_I]], align 64
// CHECK:   [[TMP28:%.*]] = load <8 x i64>, <8 x i64>* [[__A_ADDR_I10_I]], align 64
// CHECK:   [[TMP29:%.*]] = load <8 x i64>, <8 x i64>* [[__B_ADDR_I11_I]], align 64
// CHECK:   [[TMP31:%.*]] = icmp ugt <8 x i64> [[TMP28]], [[TMP29]]
// CHECK:   [[TMP32:%.*]] = select <8 x i1> [[TMP31]], <8 x i64> [[TMP28]], <8 x i64> [[TMP29]]
// CHECK:   store <8 x i64> [[TMP32]], <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP33:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP34:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[SHUFFLE6_I:%.*]] = shufflevector <8 x i64> [[TMP33]], <8 x i64> [[TMP34]], <8 x i32> <i32 0, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP35:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP36:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[SHUFFLE7_I:%.*]] = shufflevector <8 x i64> [[TMP35]], <8 x i64> [[TMP36]], <8 x i32> <i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   store <8 x i64> [[SHUFFLE6_I]], <8 x i64>* [[__A_ADDR_I_I]], align 64
// CHECK:   store <8 x i64> [[SHUFFLE7_I]], <8 x i64>* [[__B_ADDR_I_I]], align 64
// CHECK:   [[TMP37:%.*]] = load <8 x i64>, <8 x i64>* [[__A_ADDR_I_I]], align 64
// CHECK:   [[TMP38:%.*]] = load <8 x i64>, <8 x i64>* [[__B_ADDR_I_I]], align 64
// CHECK:   [[TMP40:%.*]] = icmp ugt <8 x i64> [[TMP37]], [[TMP38]]
// CHECK:   [[TMP41:%.*]] = select <8 x i1> [[TMP40]], <8 x i64> [[TMP37]], <8 x i64> [[TMP38]]
// CHECK:   store <8 x i64> [[TMP41]], <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP42:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[VECEXT_I:%.*]] = extractelement <8 x i64> [[TMP42]], i32 0
// CHECK:   ret i64 [[VECEXT_I]]
unsigned long test_mm512_mask_reduce_max_epu64(__mmask8 __M, __m512i __W){
  return _mm512_mask_reduce_max_epu64(__M, __W); 
}

// CHECK-LABEL: define i64 @test_mm512_mask_reduce_max_pd(i8 zeroext %__M, <8 x double> %__W) #0 {
// CHECK:   [[_COMPOUNDLITERAL_I_I12_I:%.*]] = alloca <8 x double>, align 64
// CHECK:   [[__A_ADDR_I13_I:%.*]] = alloca <8 x double>, align 64
// CHECK:   [[__B_ADDR_I14_I:%.*]] = alloca <8 x double>, align 64
// CHECK:   [[_COMPOUNDLITERAL_I_I9_I:%.*]] = alloca <8 x double>, align 64
// CHECK:   [[__A_ADDR_I10_I:%.*]] = alloca <8 x double>, align 64
// CHECK:   [[__B_ADDR_I11_I:%.*]] = alloca <8 x double>, align 64
// CHECK:   [[_COMPOUNDLITERAL_I_I_I:%.*]] = alloca <8 x double>, align 64
// CHECK:   [[__A_ADDR_I_I:%.*]] = alloca <8 x double>, align 64
// CHECK:   [[__B_ADDR_I_I:%.*]] = alloca <8 x double>, align 64
// CHECK:   [[__W_ADDR_I_I:%.*]] = alloca double, align 8
// CHECK:   [[_COMPOUNDLITERAL_I_I:%.*]] = alloca <8 x double>, align 64
// CHECK:   [[__M_ADDR_I:%.*]] = alloca i8, align 1
// CHECK:   [[__V_ADDR_I:%.*]] = alloca <8 x double>, align 64
// CHECK:   [[__M_ADDR:%.*]] = alloca i8, align 1
// CHECK:   [[__W_ADDR:%.*]] = alloca <8 x double>, align 64
// CHECK:   store i8 %__M, i8* [[__M_ADDR]], align 1
// CHECK:   store <8 x double> %__W, <8 x double>* [[__W_ADDR]], align 64
// CHECK:   [[TMP0:%.*]] = load i8, i8* [[__M_ADDR]], align 1
// CHECK:   [[TMP1:%.*]] = load <8 x double>, <8 x double>* [[__W_ADDR]], align 64
// CHECK:   store i8 [[TMP0]], i8* [[__M_ADDR_I]], align 1
// CHECK:   store <8 x double> [[TMP1]], <8 x double>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP2:%.*]] = load i8, i8* [[__M_ADDR_I]], align 1
// CHECK:   [[TMP3:%.*]] = load <8 x double>, <8 x double>* [[__V_ADDR_I]], align 64
// CHECK:   store double 0xFFF0000000000000, double* [[__W_ADDR_I_I]], align 8
// CHECK:   [[TMP4:%.*]] = load double, double* [[__W_ADDR_I_I]], align 8
// CHECK:   [[VECINIT_I_I:%.*]] = insertelement <8 x double> undef, double [[TMP4]], i32 0
// CHECK:   [[TMP5:%.*]] = load double, double* [[__W_ADDR_I_I]], align 8
// CHECK:   [[VECINIT1_I_I:%.*]] = insertelement <8 x double> [[VECINIT_I_I]], double [[TMP5]], i32 1
// CHECK:   [[TMP6:%.*]] = load double, double* [[__W_ADDR_I_I]], align 8
// CHECK:   [[VECINIT2_I_I:%.*]] = insertelement <8 x double> [[VECINIT1_I_I]], double [[TMP6]], i32 2
// CHECK:   [[TMP7:%.*]] = load double, double* [[__W_ADDR_I_I]], align 8
// CHECK:   [[VECINIT3_I_I:%.*]] = insertelement <8 x double> [[VECINIT2_I_I]], double [[TMP7]], i32 3
// CHECK:   [[TMP8:%.*]] = load double, double* [[__W_ADDR_I_I]], align 8
// CHECK:   [[VECINIT4_I_I:%.*]] = insertelement <8 x double> [[VECINIT3_I_I]], double [[TMP8]], i32 4
// CHECK:   [[TMP9:%.*]] = load double, double* [[__W_ADDR_I_I]], align 8
// CHECK:   [[VECINIT5_I_I:%.*]] = insertelement <8 x double> [[VECINIT4_I_I]], double [[TMP9]], i32 5
// CHECK:   [[TMP10:%.*]] = load double, double* [[__W_ADDR_I_I]], align 8
// CHECK:   [[VECINIT6_I_I:%.*]] = insertelement <8 x double> [[VECINIT5_I_I]], double [[TMP10]], i32 6
// CHECK:   [[TMP11:%.*]] = load double, double* [[__W_ADDR_I_I]], align 8
// CHECK:   [[VECINIT7_I_I:%.*]] = insertelement <8 x double> [[VECINIT6_I_I]], double [[TMP11]], i32 7
// CHECK:   store <8 x double> [[VECINIT7_I_I]], <8 x double>* [[_COMPOUNDLITERAL_I_I]], align 64
// CHECK:   [[TMP12:%.*]] = load <8 x double>, <8 x double>* [[_COMPOUNDLITERAL_I_I]], align 64
// CHECK:   [[TMP13:%.*]] = bitcast i8 [[TMP2]] to <8 x i1>
// CHECK:   [[TMP14:%.*]] = select <8 x i1> [[TMP13]], <8 x double> [[TMP3]], <8 x double> [[TMP12]]
// CHECK:   store <8 x double> [[TMP14]], <8 x double>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP15:%.*]] = load <8 x double>, <8 x double>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP16:%.*]] = load <8 x double>, <8 x double>* [[__V_ADDR_I]], align 64
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x double> [[TMP15]], <8 x double> [[TMP16]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP17:%.*]] = load <8 x double>, <8 x double>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP18:%.*]] = load <8 x double>, <8 x double>* [[__V_ADDR_I]], align 64
// CHECK:   [[SHUFFLE1_I:%.*]] = shufflevector <8 x double> [[TMP17]], <8 x double> [[TMP18]], <8 x i32> <i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   store <8 x double> [[SHUFFLE_I]], <8 x double>* [[__A_ADDR_I13_I]], align 64
// CHECK:   store <8 x double> [[SHUFFLE1_I]], <8 x double>* [[__B_ADDR_I14_I]], align 64
// CHECK:   [[TMP19:%.*]] = load <8 x double>, <8 x double>* [[__A_ADDR_I13_I]], align 64
// CHECK:   [[TMP20:%.*]] = load <8 x double>, <8 x double>* [[__B_ADDR_I14_I]], align 64
// CHECK:   store <8 x double> zeroinitializer, <8 x double>* [[_COMPOUNDLITERAL_I_I12_I]], align 64
// CHECK:   [[TMP21:%.*]] = load <8 x double>, <8 x double>* [[_COMPOUNDLITERAL_I_I12_I]], align 64
// CHECK:   [[TMP22:%.*]] = call <8 x double> @llvm.x86.avx512.mask.max.pd.512(<8 x double> [[TMP19]], <8 x double> [[TMP20]], <8 x double> [[TMP21]], i8 -1, i32 4) #2
// CHECK:   store <8 x double> [[TMP22]], <8 x double>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP23:%.*]] = load <8 x double>, <8 x double>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP24:%.*]] = load <8 x double>, <8 x double>* [[__V_ADDR_I]], align 64
// CHECK:   [[SHUFFLE3_I:%.*]] = shufflevector <8 x double> [[TMP23]], <8 x double> [[TMP24]], <8 x i32> <i32 0, i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP25:%.*]] = load <8 x double>, <8 x double>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP26:%.*]] = load <8 x double>, <8 x double>* [[__V_ADDR_I]], align 64
// CHECK:   [[SHUFFLE4_I:%.*]] = shufflevector <8 x double> [[TMP25]], <8 x double> [[TMP26]], <8 x i32> <i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   store <8 x double> [[SHUFFLE3_I]], <8 x double>* [[__A_ADDR_I10_I]], align 64
// CHECK:   store <8 x double> [[SHUFFLE4_I]], <8 x double>* [[__B_ADDR_I11_I]], align 64
// CHECK:   [[TMP27:%.*]] = load <8 x double>, <8 x double>* [[__A_ADDR_I10_I]], align 64
// CHECK:   [[TMP28:%.*]] = load <8 x double>, <8 x double>* [[__B_ADDR_I11_I]], align 64
// CHECK:   store <8 x double> zeroinitializer, <8 x double>* [[_COMPOUNDLITERAL_I_I9_I]], align 64
// CHECK:   [[TMP29:%.*]] = load <8 x double>, <8 x double>* [[_COMPOUNDLITERAL_I_I9_I]], align 64
// CHECK:   [[TMP30:%.*]] = call <8 x double> @llvm.x86.avx512.mask.max.pd.512(<8 x double> [[TMP27]], <8 x double> [[TMP28]], <8 x double> [[TMP29]], i8 -1, i32 4) #2
// CHECK:   store <8 x double> [[TMP30]], <8 x double>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP31:%.*]] = load <8 x double>, <8 x double>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP32:%.*]] = load <8 x double>, <8 x double>* [[__V_ADDR_I]], align 64
// CHECK:   [[SHUFFLE6_I:%.*]] = shufflevector <8 x double> [[TMP31]], <8 x double> [[TMP32]], <8 x i32> <i32 0, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP33:%.*]] = load <8 x double>, <8 x double>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP34:%.*]] = load <8 x double>, <8 x double>* [[__V_ADDR_I]], align 64
// CHECK:   [[SHUFFLE7_I:%.*]] = shufflevector <8 x double> [[TMP33]], <8 x double> [[TMP34]], <8 x i32> <i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   store <8 x double> [[SHUFFLE6_I]], <8 x double>* [[__A_ADDR_I_I]], align 64
// CHECK:   store <8 x double> [[SHUFFLE7_I]], <8 x double>* [[__B_ADDR_I_I]], align 64
// CHECK:   [[TMP35:%.*]] = load <8 x double>, <8 x double>* [[__A_ADDR_I_I]], align 64
// CHECK:   [[TMP36:%.*]] = load <8 x double>, <8 x double>* [[__B_ADDR_I_I]], align 64
// CHECK:   store <8 x double> zeroinitializer, <8 x double>* [[_COMPOUNDLITERAL_I_I_I]], align 64
// CHECK:   [[TMP37:%.*]] = load <8 x double>, <8 x double>* [[_COMPOUNDLITERAL_I_I_I]], align 64
// CHECK:   [[TMP38:%.*]] = call <8 x double> @llvm.x86.avx512.mask.max.pd.512(<8 x double> [[TMP35]], <8 x double> [[TMP36]], <8 x double> [[TMP37]], i8 -1, i32 4) #2
// CHECK:   store <8 x double> [[TMP38]], <8 x double>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP39:%.*]] = load <8 x double>, <8 x double>* [[__V_ADDR_I]], align 64
// CHECK:   [[VECEXT_I:%.*]] = extractelement <8 x double> [[TMP39]], i32 0
// CHECK:   [[CONV:%.*]] = fptosi double [[VECEXT_I]] to i64
// CHECK:   ret i64 [[CONV]]
long long test_mm512_mask_reduce_max_pd(__mmask8 __M, __m512d __W){
  return _mm512_mask_reduce_max_pd(__M, __W); 
}

// CHECK-LABEL: define i64 @test_mm512_mask_reduce_min_epi64(i8 zeroext %__M, <8 x i64> %__W) #0 {
// CHECK:   [[__A_ADDR_I13_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__B_ADDR_I14_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__A_ADDR_I10_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__B_ADDR_I11_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__A_ADDR_I_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__B_ADDR_I_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__D_ADDR_I_I:%.*]] = alloca i64, align 8
// CHECK:   [[_COMPOUNDLITERAL_I_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__M_ADDR_I:%.*]] = alloca i8, align 1
// CHECK:   [[__V_ADDR_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__M_ADDR:%.*]] = alloca i8, align 1
// CHECK:   [[__W_ADDR:%.*]] = alloca <8 x i64>, align 64
// CHECK:   store i8 %__M, i8* [[__M_ADDR]], align 1
// CHECK:   store <8 x i64> %__W, <8 x i64>* [[__W_ADDR]], align 64
// CHECK:   [[TMP0:%.*]] = load i8, i8* [[__M_ADDR]], align 1
// CHECK:   [[TMP1:%.*]] = load <8 x i64>, <8 x i64>* [[__W_ADDR]], align 64
// CHECK:   store i8 [[TMP0]], i8* [[__M_ADDR_I]], align 1
// CHECK:   store <8 x i64> [[TMP1]], <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP2:%.*]] = load i8, i8* [[__M_ADDR_I]], align 1
// CHECK:   [[TMP3:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   store i64 9223372036854775807, i64* [[__D_ADDR_I_I]], align 8
// CHECK:   [[TMP4:%.*]] = load i64, i64* [[__D_ADDR_I_I]], align 8
// CHECK:   [[VECINIT_I_I:%.*]] = insertelement <8 x i64> undef, i64 [[TMP4]], i32 0
// CHECK:   [[TMP5:%.*]] = load i64, i64* [[__D_ADDR_I_I]], align 8
// CHECK:   [[VECINIT1_I_I:%.*]] = insertelement <8 x i64> [[VECINIT_I_I]], i64 [[TMP5]], i32 1
// CHECK:   [[TMP6:%.*]] = load i64, i64* [[__D_ADDR_I_I]], align 8
// CHECK:   [[VECINIT2_I_I:%.*]] = insertelement <8 x i64> [[VECINIT1_I_I]], i64 [[TMP6]], i32 2
// CHECK:   [[TMP7:%.*]] = load i64, i64* [[__D_ADDR_I_I]], align 8
// CHECK:   [[VECINIT3_I_I:%.*]] = insertelement <8 x i64> [[VECINIT2_I_I]], i64 [[TMP7]], i32 3
// CHECK:   [[TMP8:%.*]] = load i64, i64* [[__D_ADDR_I_I]], align 8
// CHECK:   [[VECINIT4_I_I:%.*]] = insertelement <8 x i64> [[VECINIT3_I_I]], i64 [[TMP8]], i32 4
// CHECK:   [[TMP9:%.*]] = load i64, i64* [[__D_ADDR_I_I]], align 8
// CHECK:   [[VECINIT5_I_I:%.*]] = insertelement <8 x i64> [[VECINIT4_I_I]], i64 [[TMP9]], i32 5
// CHECK:   [[TMP10:%.*]] = load i64, i64* [[__D_ADDR_I_I]], align 8
// CHECK:   [[VECINIT6_I_I:%.*]] = insertelement <8 x i64> [[VECINIT5_I_I]], i64 [[TMP10]], i32 6
// CHECK:   [[TMP11:%.*]] = load i64, i64* [[__D_ADDR_I_I]], align 8
// CHECK:   [[VECINIT7_I_I:%.*]] = insertelement <8 x i64> [[VECINIT6_I_I]], i64 [[TMP11]], i32 7
// CHECK:   store <8 x i64> [[VECINIT7_I_I]], <8 x i64>* [[_COMPOUNDLITERAL_I_I]], align 64
// CHECK:   [[TMP12:%.*]] = load <8 x i64>, <8 x i64>* [[_COMPOUNDLITERAL_I_I]], align 64
// CHECK:   [[TMP13:%.*]] = bitcast i8 [[TMP2]] to <8 x i1>
// CHECK:   [[TMP14:%.*]] = select <8 x i1> [[TMP13]], <8 x i64> [[TMP3]], <8 x i64> [[TMP12]]
// CHECK:   store <8 x i64> [[TMP14]], <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP15:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP16:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i64> [[TMP15]], <8 x i64> [[TMP16]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP17:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP18:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[SHUFFLE1_I:%.*]] = shufflevector <8 x i64> [[TMP17]], <8 x i64> [[TMP18]], <8 x i32> <i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   store <8 x i64> [[SHUFFLE_I]], <8 x i64>* [[__A_ADDR_I13_I]], align 64
// CHECK:   store <8 x i64> [[SHUFFLE1_I]], <8 x i64>* [[__B_ADDR_I14_I]], align 64
// CHECK:   [[TMP19:%.*]] = load <8 x i64>, <8 x i64>* [[__A_ADDR_I13_I]], align 64
// CHECK:   [[TMP20:%.*]] = load <8 x i64>, <8 x i64>* [[__B_ADDR_I14_I]], align 64
// CHECK:   [[TMP22:%.*]] = icmp slt <8 x i64> [[TMP19]], [[TMP20]]
// CHECK:   [[TMP23:%.*]] = select <8 x i1> [[TMP22]], <8 x i64> [[TMP19]], <8 x i64> [[TMP20]]
// CHECK:   store <8 x i64> [[TMP23]], <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP24:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP25:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[SHUFFLE3_I:%.*]] = shufflevector <8 x i64> [[TMP24]], <8 x i64> [[TMP25]], <8 x i32> <i32 0, i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP26:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP27:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[SHUFFLE4_I:%.*]] = shufflevector <8 x i64> [[TMP26]], <8 x i64> [[TMP27]], <8 x i32> <i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   store <8 x i64> [[SHUFFLE3_I]], <8 x i64>* [[__A_ADDR_I10_I]], align 64
// CHECK:   store <8 x i64> [[SHUFFLE4_I]], <8 x i64>* [[__B_ADDR_I11_I]], align 64
// CHECK:   [[TMP28:%.*]] = load <8 x i64>, <8 x i64>* [[__A_ADDR_I10_I]], align 64
// CHECK:   [[TMP29:%.*]] = load <8 x i64>, <8 x i64>* [[__B_ADDR_I11_I]], align 64
// CHECK:   [[TMP31:%.*]] = icmp slt <8 x i64> [[TMP28]], [[TMP29]]
// CHECK:   [[TMP32:%.*]] = select <8 x i1> [[TMP31]], <8 x i64> [[TMP28]], <8 x i64> [[TMP29]]
// CHECK:   store <8 x i64> [[TMP32]], <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP33:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP34:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[SHUFFLE6_I:%.*]] = shufflevector <8 x i64> [[TMP33]], <8 x i64> [[TMP34]], <8 x i32> <i32 0, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP35:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP36:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[SHUFFLE7_I:%.*]] = shufflevector <8 x i64> [[TMP35]], <8 x i64> [[TMP36]], <8 x i32> <i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   store <8 x i64> [[SHUFFLE6_I]], <8 x i64>* [[__A_ADDR_I_I]], align 64
// CHECK:   store <8 x i64> [[SHUFFLE7_I]], <8 x i64>* [[__B_ADDR_I_I]], align 64
// CHECK:   [[TMP37:%.*]] = load <8 x i64>, <8 x i64>* [[__A_ADDR_I_I]], align 64
// CHECK:   [[TMP38:%.*]] = load <8 x i64>, <8 x i64>* [[__B_ADDR_I_I]], align 64
// CHECK:   [[TMP40:%.*]] = icmp slt <8 x i64> [[TMP37]], [[TMP38]]
// CHECK:   [[TMP41:%.*]] = select <8 x i1> [[TMP40]], <8 x i64> [[TMP37]], <8 x i64> [[TMP38]]
// CHECK:   store <8 x i64> [[TMP41]], <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP42:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[VECEXT_I:%.*]] = extractelement <8 x i64> [[TMP42]], i32 0
// CHECK:   ret i64 [[VECEXT_I]]
long long test_mm512_mask_reduce_min_epi64(__mmask8 __M, __m512i __W){
  return _mm512_mask_reduce_min_epi64(__M, __W); 
}

// CHECK-LABEL: define i64 @test_mm512_mask_reduce_min_epu64(i8 zeroext %__M, <8 x i64> %__W) #0 {
// CHECK:   [[__A_ADDR_I13_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__B_ADDR_I14_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__A_ADDR_I10_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__B_ADDR_I11_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__A_ADDR_I_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__B_ADDR_I_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__D_ADDR_I_I:%.*]] = alloca i64, align 8
// CHECK:   [[_COMPOUNDLITERAL_I_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__M_ADDR_I:%.*]] = alloca i8, align 1
// CHECK:   [[__V_ADDR_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__M_ADDR:%.*]] = alloca i8, align 1
// CHECK:   [[__W_ADDR:%.*]] = alloca <8 x i64>, align 64
// CHECK:   store i8 %__M, i8* [[__M_ADDR]], align 1
// CHECK:   store <8 x i64> %__W, <8 x i64>* [[__W_ADDR]], align 64
// CHECK:   [[TMP0:%.*]] = load i8, i8* [[__M_ADDR]], align 1
// CHECK:   [[TMP1:%.*]] = load <8 x i64>, <8 x i64>* [[__W_ADDR]], align 64
// CHECK:   store i8 [[TMP0]], i8* [[__M_ADDR_I]], align 1
// CHECK:   store <8 x i64> [[TMP1]], <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP2:%.*]] = load i8, i8* [[__M_ADDR_I]], align 1
// CHECK:   [[TMP3:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   store i64 -1, i64* [[__D_ADDR_I_I]], align 8
// CHECK:   [[TMP4:%.*]] = load i64, i64* [[__D_ADDR_I_I]], align 8
// CHECK:   [[VECINIT_I_I:%.*]] = insertelement <8 x i64> undef, i64 [[TMP4]], i32 0
// CHECK:   [[TMP5:%.*]] = load i64, i64* [[__D_ADDR_I_I]], align 8
// CHECK:   [[VECINIT1_I_I:%.*]] = insertelement <8 x i64> [[VECINIT_I_I]], i64 [[TMP5]], i32 1
// CHECK:   [[TMP6:%.*]] = load i64, i64* [[__D_ADDR_I_I]], align 8
// CHECK:   [[VECINIT2_I_I:%.*]] = insertelement <8 x i64> [[VECINIT1_I_I]], i64 [[TMP6]], i32 2
// CHECK:   [[TMP7:%.*]] = load i64, i64* [[__D_ADDR_I_I]], align 8
// CHECK:   [[VECINIT3_I_I:%.*]] = insertelement <8 x i64> [[VECINIT2_I_I]], i64 [[TMP7]], i32 3
// CHECK:   [[TMP8:%.*]] = load i64, i64* [[__D_ADDR_I_I]], align 8
// CHECK:   [[VECINIT4_I_I:%.*]] = insertelement <8 x i64> [[VECINIT3_I_I]], i64 [[TMP8]], i32 4
// CHECK:   [[TMP9:%.*]] = load i64, i64* [[__D_ADDR_I_I]], align 8
// CHECK:   [[VECINIT5_I_I:%.*]] = insertelement <8 x i64> [[VECINIT4_I_I]], i64 [[TMP9]], i32 5
// CHECK:   [[TMP10:%.*]] = load i64, i64* [[__D_ADDR_I_I]], align 8
// CHECK:   [[VECINIT6_I_I:%.*]] = insertelement <8 x i64> [[VECINIT5_I_I]], i64 [[TMP10]], i32 6
// CHECK:   [[TMP11:%.*]] = load i64, i64* [[__D_ADDR_I_I]], align 8
// CHECK:   [[VECINIT7_I_I:%.*]] = insertelement <8 x i64> [[VECINIT6_I_I]], i64 [[TMP11]], i32 7
// CHECK:   store <8 x i64> [[VECINIT7_I_I]], <8 x i64>* [[_COMPOUNDLITERAL_I_I]], align 64
// CHECK:   [[TMP12:%.*]] = load <8 x i64>, <8 x i64>* [[_COMPOUNDLITERAL_I_I]], align 64
// CHECK:   [[TMP13:%.*]] = bitcast i8 [[TMP2]] to <8 x i1>
// CHECK:   [[TMP14:%.*]] = select <8 x i1> [[TMP13]], <8 x i64> [[TMP3]], <8 x i64> [[TMP12]]
// CHECK:   store <8 x i64> [[TMP14]], <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP15:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP16:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i64> [[TMP15]], <8 x i64> [[TMP16]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP17:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP18:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[SHUFFLE1_I:%.*]] = shufflevector <8 x i64> [[TMP17]], <8 x i64> [[TMP18]], <8 x i32> <i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   store <8 x i64> [[SHUFFLE_I]], <8 x i64>* [[__A_ADDR_I13_I]], align 64
// CHECK:   store <8 x i64> [[SHUFFLE1_I]], <8 x i64>* [[__B_ADDR_I14_I]], align 64
// CHECK:   [[TMP19:%.*]] = load <8 x i64>, <8 x i64>* [[__A_ADDR_I13_I]], align 64
// CHECK:   [[TMP20:%.*]] = load <8 x i64>, <8 x i64>* [[__B_ADDR_I14_I]], align 64
// CHECK:   [[TMP22:%.*]] = icmp ult <8 x i64> [[TMP19]], [[TMP20]]
// CHECK:   [[TMP23:%.*]] = select <8 x i1> [[TMP22]], <8 x i64> [[TMP19]], <8 x i64> [[TMP20]]
// CHECK:   store <8 x i64> [[TMP23]], <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP24:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP25:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[SHUFFLE3_I:%.*]] = shufflevector <8 x i64> [[TMP24]], <8 x i64> [[TMP25]], <8 x i32> <i32 0, i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP26:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP27:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[SHUFFLE4_I:%.*]] = shufflevector <8 x i64> [[TMP26]], <8 x i64> [[TMP27]], <8 x i32> <i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   store <8 x i64> [[SHUFFLE3_I]], <8 x i64>* [[__A_ADDR_I10_I]], align 64
// CHECK:   store <8 x i64> [[SHUFFLE4_I]], <8 x i64>* [[__B_ADDR_I11_I]], align 64
// CHECK:   [[TMP28:%.*]] = load <8 x i64>, <8 x i64>* [[__A_ADDR_I10_I]], align 64
// CHECK:   [[TMP29:%.*]] = load <8 x i64>, <8 x i64>* [[__B_ADDR_I11_I]], align 64
// CHECK:   [[TMP31:%.*]] = icmp ult <8 x i64> [[TMP28]], [[TMP29]]
// CHECK:   [[TMP32:%.*]] = select <8 x i1> [[TMP31]], <8 x i64> [[TMP28]], <8 x i64> [[TMP29]]
// CHECK:   store <8 x i64> [[TMP32]], <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP33:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP34:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[SHUFFLE6_I:%.*]] = shufflevector <8 x i64> [[TMP33]], <8 x i64> [[TMP34]], <8 x i32> <i32 0, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP35:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP36:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[SHUFFLE7_I:%.*]] = shufflevector <8 x i64> [[TMP35]], <8 x i64> [[TMP36]], <8 x i32> <i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   store <8 x i64> [[SHUFFLE6_I]], <8 x i64>* [[__A_ADDR_I_I]], align 64
// CHECK:   store <8 x i64> [[SHUFFLE7_I]], <8 x i64>* [[__B_ADDR_I_I]], align 64
// CHECK:   [[TMP37:%.*]] = load <8 x i64>, <8 x i64>* [[__A_ADDR_I_I]], align 64
// CHECK:   [[TMP38:%.*]] = load <8 x i64>, <8 x i64>* [[__B_ADDR_I_I]], align 64
// CHECK:   [[TMP40:%.*]] = icmp ult <8 x i64> [[TMP37]], [[TMP38]]
// CHECK:   [[TMP41:%.*]] = select <8 x i1> [[TMP40]], <8 x i64> [[TMP37]], <8 x i64> [[TMP38]]
// CHECK:   store <8 x i64> [[TMP41]], <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP42:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[VECEXT_I:%.*]] = extractelement <8 x i64> [[TMP42]], i32 0
// CHECK:   ret i64 [[VECEXT_I]]
long long test_mm512_mask_reduce_min_epu64(__mmask8 __M, __m512i __W){
  return _mm512_mask_reduce_min_epu64(__M, __W);
}

// CHECK-LABEL: define double @test_mm512_mask_reduce_min_pd(i8 zeroext %__M, <8 x double> %__W) #0 {
// CHECK:   [[_COMPOUNDLITERAL_I_I12_I:%.*]] = alloca <8 x double>, align 64
// CHECK:   [[__A_ADDR_I13_I:%.*]] = alloca <8 x double>, align 64
// CHECK:   [[__B_ADDR_I14_I:%.*]] = alloca <8 x double>, align 64
// CHECK:   [[_COMPOUNDLITERAL_I_I9_I:%.*]] = alloca <8 x double>, align 64
// CHECK:   [[__A_ADDR_I10_I:%.*]] = alloca <8 x double>, align 64
// CHECK:   [[__B_ADDR_I11_I:%.*]] = alloca <8 x double>, align 64
// CHECK:   [[_COMPOUNDLITERAL_I_I_I:%.*]] = alloca <8 x double>, align 64
// CHECK:   [[__A_ADDR_I_I:%.*]] = alloca <8 x double>, align 64
// CHECK:   [[__B_ADDR_I_I:%.*]] = alloca <8 x double>, align 64
// CHECK:   [[__W_ADDR_I_I:%.*]] = alloca double, align 8
// CHECK:   [[_COMPOUNDLITERAL_I_I:%.*]] = alloca <8 x double>, align 64
// CHECK:   [[__M_ADDR_I:%.*]] = alloca i8, align 1
// CHECK:   [[__V_ADDR_I:%.*]] = alloca <8 x double>, align 64
// CHECK:   [[__M_ADDR:%.*]] = alloca i8, align 1
// CHECK:   [[__W_ADDR:%.*]] = alloca <8 x double>, align 64
// CHECK:   store i8 %__M, i8* [[__M_ADDR]], align 1
// CHECK:   store <8 x double> %__W, <8 x double>* [[__W_ADDR]], align 64
// CHECK:   [[TMP0:%.*]] = load i8, i8* [[__M_ADDR]], align 1
// CHECK:   [[TMP1:%.*]] = load <8 x double>, <8 x double>* [[__W_ADDR]], align 64
// CHECK:   store i8 [[TMP0]], i8* [[__M_ADDR_I]], align 1
// CHECK:   store <8 x double> [[TMP1]], <8 x double>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP2:%.*]] = load i8, i8* [[__M_ADDR_I]], align 1
// CHECK:   [[TMP3:%.*]] = load <8 x double>, <8 x double>* [[__V_ADDR_I]], align 64
// CHECK:   store double 0x7FF0000000000000, double* [[__W_ADDR_I_I]], align 8
// CHECK:   [[TMP4:%.*]] = load double, double* [[__W_ADDR_I_I]], align 8
// CHECK:   [[VECINIT_I_I:%.*]] = insertelement <8 x double> undef, double [[TMP4]], i32 0
// CHECK:   [[TMP5:%.*]] = load double, double* [[__W_ADDR_I_I]], align 8
// CHECK:   [[VECINIT1_I_I:%.*]] = insertelement <8 x double> [[VECINIT_I_I]], double [[TMP5]], i32 1
// CHECK:   [[TMP6:%.*]] = load double, double* [[__W_ADDR_I_I]], align 8
// CHECK:   [[VECINIT2_I_I:%.*]] = insertelement <8 x double> [[VECINIT1_I_I]], double [[TMP6]], i32 2
// CHECK:   [[TMP7:%.*]] = load double, double* [[__W_ADDR_I_I]], align 8
// CHECK:   [[VECINIT3_I_I:%.*]] = insertelement <8 x double> [[VECINIT2_I_I]], double [[TMP7]], i32 3
// CHECK:   [[TMP8:%.*]] = load double, double* [[__W_ADDR_I_I]], align 8
// CHECK:   [[VECINIT4_I_I:%.*]] = insertelement <8 x double> [[VECINIT3_I_I]], double [[TMP8]], i32 4
// CHECK:   [[TMP9:%.*]] = load double, double* [[__W_ADDR_I_I]], align 8
// CHECK:   [[VECINIT5_I_I:%.*]] = insertelement <8 x double> [[VECINIT4_I_I]], double [[TMP9]], i32 5
// CHECK:   [[TMP10:%.*]] = load double, double* [[__W_ADDR_I_I]], align 8
// CHECK:   [[VECINIT6_I_I:%.*]] = insertelement <8 x double> [[VECINIT5_I_I]], double [[TMP10]], i32 6
// CHECK:   [[TMP11:%.*]] = load double, double* [[__W_ADDR_I_I]], align 8
// CHECK:   [[VECINIT7_I_I:%.*]] = insertelement <8 x double> [[VECINIT6_I_I]], double [[TMP11]], i32 7
// CHECK:   store <8 x double> [[VECINIT7_I_I]], <8 x double>* [[_COMPOUNDLITERAL_I_I]], align 64
// CHECK:   [[TMP12:%.*]] = load <8 x double>, <8 x double>* [[_COMPOUNDLITERAL_I_I]], align 64
// CHECK:   [[TMP13:%.*]] = bitcast i8 [[TMP2]] to <8 x i1>
// CHECK:   [[TMP14:%.*]] = select <8 x i1> [[TMP13]], <8 x double> [[TMP3]], <8 x double> [[TMP12]]
// CHECK:   store <8 x double> [[TMP14]], <8 x double>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP15:%.*]] = load <8 x double>, <8 x double>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP16:%.*]] = load <8 x double>, <8 x double>* [[__V_ADDR_I]], align 64
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x double> [[TMP15]], <8 x double> [[TMP16]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP17:%.*]] = load <8 x double>, <8 x double>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP18:%.*]] = load <8 x double>, <8 x double>* [[__V_ADDR_I]], align 64
// CHECK:   [[SHUFFLE1_I:%.*]] = shufflevector <8 x double> [[TMP17]], <8 x double> [[TMP18]], <8 x i32> <i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   store <8 x double> [[SHUFFLE_I]], <8 x double>* [[__A_ADDR_I13_I]], align 64
// CHECK:   store <8 x double> [[SHUFFLE1_I]], <8 x double>* [[__B_ADDR_I14_I]], align 64
// CHECK:   [[TMP19:%.*]] = load <8 x double>, <8 x double>* [[__A_ADDR_I13_I]], align 64
// CHECK:   [[TMP20:%.*]] = load <8 x double>, <8 x double>* [[__B_ADDR_I14_I]], align 64
// CHECK:   store <8 x double> zeroinitializer, <8 x double>* [[_COMPOUNDLITERAL_I_I12_I]], align 64
// CHECK:   [[TMP21:%.*]] = load <8 x double>, <8 x double>* [[_COMPOUNDLITERAL_I_I12_I]], align 64
// CHECK:   [[TMP22:%.*]] = call <8 x double> @llvm.x86.avx512.mask.min.pd.512(<8 x double> [[TMP19]], <8 x double> [[TMP20]], <8 x double> [[TMP21]], i8 -1, i32 4) #2
// CHECK:   store <8 x double> [[TMP22]], <8 x double>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP23:%.*]] = load <8 x double>, <8 x double>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP24:%.*]] = load <8 x double>, <8 x double>* [[__V_ADDR_I]], align 64
// CHECK:   [[SHUFFLE3_I:%.*]] = shufflevector <8 x double> [[TMP23]], <8 x double> [[TMP24]], <8 x i32> <i32 0, i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP25:%.*]] = load <8 x double>, <8 x double>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP26:%.*]] = load <8 x double>, <8 x double>* [[__V_ADDR_I]], align 64
// CHECK:   [[SHUFFLE4_I:%.*]] = shufflevector <8 x double> [[TMP25]], <8 x double> [[TMP26]], <8 x i32> <i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   store <8 x double> [[SHUFFLE3_I]], <8 x double>* [[__A_ADDR_I10_I]], align 64
// CHECK:   store <8 x double> [[SHUFFLE4_I]], <8 x double>* [[__B_ADDR_I11_I]], align 64
// CHECK:   [[TMP27:%.*]] = load <8 x double>, <8 x double>* [[__A_ADDR_I10_I]], align 64
// CHECK:   [[TMP28:%.*]] = load <8 x double>, <8 x double>* [[__B_ADDR_I11_I]], align 64
// CHECK:   store <8 x double> zeroinitializer, <8 x double>* [[_COMPOUNDLITERAL_I_I9_I]], align 64
// CHECK:   [[TMP29:%.*]] = load <8 x double>, <8 x double>* [[_COMPOUNDLITERAL_I_I9_I]], align 64
// CHECK:   [[TMP30:%.*]] = call <8 x double> @llvm.x86.avx512.mask.min.pd.512(<8 x double> [[TMP27]], <8 x double> [[TMP28]], <8 x double> [[TMP29]], i8 -1, i32 4) #2
// CHECK:   store <8 x double> [[TMP30]], <8 x double>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP31:%.*]] = load <8 x double>, <8 x double>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP32:%.*]] = load <8 x double>, <8 x double>* [[__V_ADDR_I]], align 64
// CHECK:   [[SHUFFLE6_I:%.*]] = shufflevector <8 x double> [[TMP31]], <8 x double> [[TMP32]], <8 x i32> <i32 0, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP33:%.*]] = load <8 x double>, <8 x double>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP34:%.*]] = load <8 x double>, <8 x double>* [[__V_ADDR_I]], align 64
// CHECK:   [[SHUFFLE7_I:%.*]] = shufflevector <8 x double> [[TMP33]], <8 x double> [[TMP34]], <8 x i32> <i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   store <8 x double> [[SHUFFLE6_I]], <8 x double>* [[__A_ADDR_I_I]], align 64
// CHECK:   store <8 x double> [[SHUFFLE7_I]], <8 x double>* [[__B_ADDR_I_I]], align 64
// CHECK:   [[TMP35:%.*]] = load <8 x double>, <8 x double>* [[__A_ADDR_I_I]], align 64
// CHECK:   [[TMP36:%.*]] = load <8 x double>, <8 x double>* [[__B_ADDR_I_I]], align 64
// CHECK:   store <8 x double> zeroinitializer, <8 x double>* [[_COMPOUNDLITERAL_I_I_I]], align 64
// CHECK:   [[TMP37:%.*]] = load <8 x double>, <8 x double>* [[_COMPOUNDLITERAL_I_I_I]], align 64
// CHECK:   [[TMP38:%.*]] = call <8 x double> @llvm.x86.avx512.mask.min.pd.512(<8 x double> [[TMP35]], <8 x double> [[TMP36]], <8 x double> [[TMP37]], i8 -1, i32 4) #2
// CHECK:   store <8 x double> [[TMP38]], <8 x double>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP39:%.*]] = load <8 x double>, <8 x double>* [[__V_ADDR_I]], align 64
// CHECK:   [[VECEXT_I:%.*]] = extractelement <8 x double> [[TMP39]], i32 0
// CHECK:   ret double [[VECEXT_I]]
double test_mm512_mask_reduce_min_pd(__mmask8 __M, __m512d __W){
  return _mm512_mask_reduce_min_pd(__M, __W); 
}

// CHECK-LABEL: define i32 @test_mm512_reduce_max_epi32(<8 x i64> %__W) #0 {
// CHECK:   [[__A_ADDR_I18_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__B_ADDR_I19_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__A_ADDR_I15_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__B_ADDR_I16_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__A_ADDR_I12_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__B_ADDR_I13_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__A_ADDR_I_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__B_ADDR_I_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[A_ADDR_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__W_ADDR:%.*]] = alloca <8 x i64>, align 64
// CHECK:   store <8 x i64> %__W, <8 x i64>* [[__W_ADDR]], align 64
// CHECK:   [[TMP0:%.*]] = load <8 x i64>, <8 x i64>* [[__W_ADDR]], align 64
// CHECK:   store <8 x i64> [[TMP0]], <8 x i64>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP1:%.*]] = load <8 x i64>, <8 x i64>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP2:%.*]] = bitcast <8 x i64> [[TMP1]] to <16 x i32>
// CHECK:   [[TMP3:%.*]] = load <8 x i64>, <8 x i64>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP4:%.*]] = bitcast <8 x i64> [[TMP3]] to <16 x i32>
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <16 x i32> [[TMP2]], <16 x i32> [[TMP4]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP5:%.*]] = bitcast <16 x i32> [[SHUFFLE_I]] to <8 x i64>
// CHECK:   [[TMP6:%.*]] = load <8 x i64>, <8 x i64>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP7:%.*]] = bitcast <8 x i64> [[TMP6]] to <16 x i32>
// CHECK:   [[TMP8:%.*]] = load <8 x i64>, <8 x i64>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP9:%.*]] = bitcast <8 x i64> [[TMP8]] to <16 x i32>
// CHECK:   [[SHUFFLE1_I:%.*]] = shufflevector <16 x i32> [[TMP7]], <16 x i32> [[TMP9]], <16 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP10:%.*]] = bitcast <16 x i32> [[SHUFFLE1_I]] to <8 x i64>
// CHECK:   store <8 x i64> [[TMP5]], <8 x i64>* [[__A_ADDR_I_I]], align 64
// CHECK:   store <8 x i64> [[TMP10]], <8 x i64>* [[__B_ADDR_I_I]], align 64
// CHECK:   [[TMP11:%.*]] = load <8 x i64>, <8 x i64>* [[__A_ADDR_I_I]], align 64
// CHECK:   [[TMP12:%.*]] = bitcast <8 x i64> [[TMP11]] to <16 x i32>
// CHECK:   [[TMP13:%.*]] = load <8 x i64>, <8 x i64>* [[__B_ADDR_I_I]], align 64
// CHECK:   [[TMP14:%.*]] = bitcast <8 x i64> [[TMP13]] to <16 x i32>
// CHECK:   [[TMP17:%.*]] = icmp sgt <16 x i32> [[TMP12]], [[TMP14]]
// CHECK:   [[TMP18:%.*]] = select <16 x i1> [[TMP17]], <16 x i32> [[TMP12]], <16 x i32> [[TMP14]]
// CHECK:   [[TMP19:%.*]] = bitcast <16 x i32> [[TMP18]] to <8 x i64>
// CHECK:   store <8 x i64> [[TMP19]], <8 x i64>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP20:%.*]] = load <8 x i64>, <8 x i64>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP21:%.*]] = bitcast <8 x i64> [[TMP20]] to <16 x i32>
// CHECK:   [[TMP22:%.*]] = load <8 x i64>, <8 x i64>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP23:%.*]] = bitcast <8 x i64> [[TMP22]] to <16 x i32>
// CHECK:   [[SHUFFLE2_I:%.*]] = shufflevector <16 x i32> [[TMP21]], <16 x i32> [[TMP23]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP24:%.*]] = bitcast <16 x i32> [[SHUFFLE2_I]] to <8 x i64>
// CHECK:   [[TMP25:%.*]] = load <8 x i64>, <8 x i64>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP26:%.*]] = bitcast <8 x i64> [[TMP25]] to <16 x i32>
// CHECK:   [[TMP27:%.*]] = load <8 x i64>, <8 x i64>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP28:%.*]] = bitcast <8 x i64> [[TMP27]] to <16 x i32>
// CHECK:   [[SHUFFLE3_I:%.*]] = shufflevector <16 x i32> [[TMP26]], <16 x i32> [[TMP28]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP29:%.*]] = bitcast <16 x i32> [[SHUFFLE3_I]] to <8 x i64>
// CHECK:   store <8 x i64> [[TMP24]], <8 x i64>* [[__A_ADDR_I18_I]], align 64
// CHECK:   store <8 x i64> [[TMP29]], <8 x i64>* [[__B_ADDR_I19_I]], align 64
// CHECK:   [[TMP30:%.*]] = load <8 x i64>, <8 x i64>* [[__A_ADDR_I18_I]], align 64
// CHECK:   [[TMP31:%.*]] = bitcast <8 x i64> [[TMP30]] to <16 x i32>
// CHECK:   [[TMP32:%.*]] = load <8 x i64>, <8 x i64>* [[__B_ADDR_I19_I]], align 64
// CHECK:   [[TMP33:%.*]] = bitcast <8 x i64> [[TMP32]] to <16 x i32>
// CHECK:   [[TMP36:%.*]] = icmp sgt <16 x i32> [[TMP31]], [[TMP33]]
// CHECK:   [[TMP37:%.*]] = select <16 x i1> [[TMP36]], <16 x i32> [[TMP31]], <16 x i32> [[TMP33]]
// CHECK:   [[TMP38:%.*]] = bitcast <16 x i32> [[TMP37]] to <8 x i64>
// CHECK:   store <8 x i64> [[TMP38]], <8 x i64>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP39:%.*]] = load <8 x i64>, <8 x i64>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP40:%.*]] = bitcast <8 x i64> [[TMP39]] to <16 x i32>
// CHECK:   [[TMP41:%.*]] = load <8 x i64>, <8 x i64>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP42:%.*]] = bitcast <8 x i64> [[TMP41]] to <16 x i32>
// CHECK:   [[SHUFFLE5_I:%.*]] = shufflevector <16 x i32> [[TMP40]], <16 x i32> [[TMP42]], <16 x i32> <i32 0, i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP43:%.*]] = bitcast <16 x i32> [[SHUFFLE5_I]] to <8 x i64>
// CHECK:   [[TMP44:%.*]] = load <8 x i64>, <8 x i64>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP45:%.*]] = bitcast <8 x i64> [[TMP44]] to <16 x i32>
// CHECK:   [[TMP46:%.*]] = load <8 x i64>, <8 x i64>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP47:%.*]] = bitcast <8 x i64> [[TMP46]] to <16 x i32>
// CHECK:   [[SHUFFLE6_I:%.*]] = shufflevector <16 x i32> [[TMP45]], <16 x i32> [[TMP47]], <16 x i32> <i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP48:%.*]] = bitcast <16 x i32> [[SHUFFLE6_I]] to <8 x i64>
// CHECK:   store <8 x i64> [[TMP43]], <8 x i64>* [[__A_ADDR_I15_I]], align 64
// CHECK:   store <8 x i64> [[TMP48]], <8 x i64>* [[__B_ADDR_I16_I]], align 64
// CHECK:   [[TMP49:%.*]] = load <8 x i64>, <8 x i64>* [[__A_ADDR_I15_I]], align 64
// CHECK:   [[TMP50:%.*]] = bitcast <8 x i64> [[TMP49]] to <16 x i32>
// CHECK:   [[TMP51:%.*]] = load <8 x i64>, <8 x i64>* [[__B_ADDR_I16_I]], align 64
// CHECK:   [[TMP52:%.*]] = bitcast <8 x i64> [[TMP51]] to <16 x i32>
// CHECK:   [[TMP55:%.*]] = icmp sgt <16 x i32> [[TMP50]], [[TMP52]]
// CHECK:   [[TMP56:%.*]] = select <16 x i1> [[TMP55]], <16 x i32> [[TMP50]], <16 x i32> [[TMP52]]
// CHECK:   [[TMP57:%.*]] = bitcast <16 x i32> [[TMP56]] to <8 x i64>
// CHECK:   store <8 x i64> [[TMP57]], <8 x i64>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP58:%.*]] = load <8 x i64>, <8 x i64>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP59:%.*]] = bitcast <8 x i64> [[TMP58]] to <16 x i32>
// CHECK:   [[TMP60:%.*]] = load <8 x i64>, <8 x i64>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP61:%.*]] = bitcast <8 x i64> [[TMP60]] to <16 x i32>
// CHECK:   [[SHUFFLE8_I:%.*]] = shufflevector <16 x i32> [[TMP59]], <16 x i32> [[TMP61]], <16 x i32> <i32 0, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP62:%.*]] = bitcast <16 x i32> [[SHUFFLE8_I]] to <8 x i64>
// CHECK:   [[TMP63:%.*]] = load <8 x i64>, <8 x i64>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP64:%.*]] = bitcast <8 x i64> [[TMP63]] to <16 x i32>
// CHECK:   [[TMP65:%.*]] = load <8 x i64>, <8 x i64>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP66:%.*]] = bitcast <8 x i64> [[TMP65]] to <16 x i32>
// CHECK:   [[SHUFFLE9_I:%.*]] = shufflevector <16 x i32> [[TMP64]], <16 x i32> [[TMP66]], <16 x i32> <i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP67:%.*]] = bitcast <16 x i32> [[SHUFFLE9_I]] to <8 x i64>
// CHECK:   store <8 x i64> [[TMP62]], <8 x i64>* [[__A_ADDR_I12_I]], align 64
// CHECK:   store <8 x i64> [[TMP67]], <8 x i64>* [[__B_ADDR_I13_I]], align 64
// CHECK:   [[TMP68:%.*]] = load <8 x i64>, <8 x i64>* [[__A_ADDR_I12_I]], align 64
// CHECK:   [[TMP69:%.*]] = bitcast <8 x i64> [[TMP68]] to <16 x i32>
// CHECK:   [[TMP70:%.*]] = load <8 x i64>, <8 x i64>* [[__B_ADDR_I13_I]], align 64
// CHECK:   [[TMP71:%.*]] = bitcast <8 x i64> [[TMP70]] to <16 x i32>
// CHECK:   [[TMP74:%.*]] = icmp sgt <16 x i32> [[TMP69]], [[TMP71]]
// CHECK:   [[TMP75:%.*]] = select <16 x i1> [[TMP74]], <16 x i32> [[TMP69]], <16 x i32> [[TMP71]]
// CHECK:   [[TMP76:%.*]] = bitcast <16 x i32> [[TMP75]] to <8 x i64>
// CHECK:   store <8 x i64> [[TMP76]], <8 x i64>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP77:%.*]] = load <8 x i64>, <8 x i64>* [[A_ADDR_I]], align 64
// CHECK:   [[VECEXT_I:%.*]] = extractelement <8 x i64> [[TMP77]], i32 0
// CHECK:   [[CONV_I:%.*]] = trunc i64 [[VECEXT_I]] to i32
// CHECK:   ret i32 [[CONV_I]]
int test_mm512_reduce_max_epi32(__m512i __W){
  return _mm512_reduce_max_epi32(__W);
}

// CHECK-LABEL: define i32 @test_mm512_reduce_max_epu32(<8 x i64> %__W) #0 {
// CHECK:   [[__A_ADDR_I18_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__B_ADDR_I19_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__A_ADDR_I15_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__B_ADDR_I16_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__A_ADDR_I12_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__B_ADDR_I13_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__A_ADDR_I_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__B_ADDR_I_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[A_ADDR_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__W_ADDR:%.*]] = alloca <8 x i64>, align 64
// CHECK:   store <8 x i64> %__W, <8 x i64>* [[__W_ADDR]], align 64
// CHECK:   [[TMP0:%.*]] = load <8 x i64>, <8 x i64>* [[__W_ADDR]], align 64
// CHECK:   store <8 x i64> [[TMP0]], <8 x i64>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP1:%.*]] = load <8 x i64>, <8 x i64>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP2:%.*]] = bitcast <8 x i64> [[TMP1]] to <16 x i32>
// CHECK:   [[TMP3:%.*]] = load <8 x i64>, <8 x i64>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP4:%.*]] = bitcast <8 x i64> [[TMP3]] to <16 x i32>
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <16 x i32> [[TMP2]], <16 x i32> [[TMP4]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP5:%.*]] = bitcast <16 x i32> [[SHUFFLE_I]] to <8 x i64>
// CHECK:   [[TMP6:%.*]] = load <8 x i64>, <8 x i64>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP7:%.*]] = bitcast <8 x i64> [[TMP6]] to <16 x i32>
// CHECK:   [[TMP8:%.*]] = load <8 x i64>, <8 x i64>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP9:%.*]] = bitcast <8 x i64> [[TMP8]] to <16 x i32>
// CHECK:   [[SHUFFLE1_I:%.*]] = shufflevector <16 x i32> [[TMP7]], <16 x i32> [[TMP9]], <16 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP10:%.*]] = bitcast <16 x i32> [[SHUFFLE1_I]] to <8 x i64>
// CHECK:   store <8 x i64> [[TMP5]], <8 x i64>* [[__A_ADDR_I_I]], align 64
// CHECK:   store <8 x i64> [[TMP10]], <8 x i64>* [[__B_ADDR_I_I]], align 64
// CHECK:   [[TMP11:%.*]] = load <8 x i64>, <8 x i64>* [[__A_ADDR_I_I]], align 64
// CHECK:   [[TMP12:%.*]] = bitcast <8 x i64> [[TMP11]] to <16 x i32>
// CHECK:   [[TMP13:%.*]] = load <8 x i64>, <8 x i64>* [[__B_ADDR_I_I]], align 64
// CHECK:   [[TMP14:%.*]] = bitcast <8 x i64> [[TMP13]] to <16 x i32>
// CHECK:   [[TMP17:%.*]] = icmp ugt <16 x i32> [[TMP12]], [[TMP14]]
// CHECK:   [[TMP18:%.*]] = select <16 x i1> [[TMP17]], <16 x i32> [[TMP12]], <16 x i32> [[TMP14]]
// CHECK:   [[TMP19:%.*]] = bitcast <16 x i32> [[TMP18]] to <8 x i64>
// CHECK:   store <8 x i64> [[TMP19]], <8 x i64>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP20:%.*]] = load <8 x i64>, <8 x i64>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP21:%.*]] = bitcast <8 x i64> [[TMP20]] to <16 x i32>
// CHECK:   [[TMP22:%.*]] = load <8 x i64>, <8 x i64>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP23:%.*]] = bitcast <8 x i64> [[TMP22]] to <16 x i32>
// CHECK:   [[SHUFFLE2_I:%.*]] = shufflevector <16 x i32> [[TMP21]], <16 x i32> [[TMP23]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP24:%.*]] = bitcast <16 x i32> [[SHUFFLE2_I]] to <8 x i64>
// CHECK:   [[TMP25:%.*]] = load <8 x i64>, <8 x i64>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP26:%.*]] = bitcast <8 x i64> [[TMP25]] to <16 x i32>
// CHECK:   [[TMP27:%.*]] = load <8 x i64>, <8 x i64>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP28:%.*]] = bitcast <8 x i64> [[TMP27]] to <16 x i32>
// CHECK:   [[SHUFFLE3_I:%.*]] = shufflevector <16 x i32> [[TMP26]], <16 x i32> [[TMP28]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP29:%.*]] = bitcast <16 x i32> [[SHUFFLE3_I]] to <8 x i64>
// CHECK:   store <8 x i64> [[TMP24]], <8 x i64>* [[__A_ADDR_I18_I]], align 64
// CHECK:   store <8 x i64> [[TMP29]], <8 x i64>* [[__B_ADDR_I19_I]], align 64
// CHECK:   [[TMP30:%.*]] = load <8 x i64>, <8 x i64>* [[__A_ADDR_I18_I]], align 64
// CHECK:   [[TMP31:%.*]] = bitcast <8 x i64> [[TMP30]] to <16 x i32>
// CHECK:   [[TMP32:%.*]] = load <8 x i64>, <8 x i64>* [[__B_ADDR_I19_I]], align 64
// CHECK:   [[TMP33:%.*]] = bitcast <8 x i64> [[TMP32]] to <16 x i32>
// CHECK:   [[TMP36:%.*]] = icmp ugt <16 x i32> [[TMP31]], [[TMP33]]
// CHECK:   [[TMP37:%.*]] = select <16 x i1> [[TMP36]], <16 x i32> [[TMP31]], <16 x i32> [[TMP33]]
// CHECK:   [[TMP38:%.*]] = bitcast <16 x i32> [[TMP37]] to <8 x i64>
// CHECK:   store <8 x i64> [[TMP38]], <8 x i64>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP39:%.*]] = load <8 x i64>, <8 x i64>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP40:%.*]] = bitcast <8 x i64> [[TMP39]] to <16 x i32>
// CHECK:   [[TMP41:%.*]] = load <8 x i64>, <8 x i64>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP42:%.*]] = bitcast <8 x i64> [[TMP41]] to <16 x i32>
// CHECK:   [[SHUFFLE5_I:%.*]] = shufflevector <16 x i32> [[TMP40]], <16 x i32> [[TMP42]], <16 x i32> <i32 0, i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP43:%.*]] = bitcast <16 x i32> [[SHUFFLE5_I]] to <8 x i64>
// CHECK:   [[TMP44:%.*]] = load <8 x i64>, <8 x i64>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP45:%.*]] = bitcast <8 x i64> [[TMP44]] to <16 x i32>
// CHECK:   [[TMP46:%.*]] = load <8 x i64>, <8 x i64>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP47:%.*]] = bitcast <8 x i64> [[TMP46]] to <16 x i32>
// CHECK:   [[SHUFFLE6_I:%.*]] = shufflevector <16 x i32> [[TMP45]], <16 x i32> [[TMP47]], <16 x i32> <i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP48:%.*]] = bitcast <16 x i32> [[SHUFFLE6_I]] to <8 x i64>
// CHECK:   store <8 x i64> [[TMP43]], <8 x i64>* [[__A_ADDR_I15_I]], align 64
// CHECK:   store <8 x i64> [[TMP48]], <8 x i64>* [[__B_ADDR_I16_I]], align 64
// CHECK:   [[TMP49:%.*]] = load <8 x i64>, <8 x i64>* [[__A_ADDR_I15_I]], align 64
// CHECK:   [[TMP50:%.*]] = bitcast <8 x i64> [[TMP49]] to <16 x i32>
// CHECK:   [[TMP51:%.*]] = load <8 x i64>, <8 x i64>* [[__B_ADDR_I16_I]], align 64
// CHECK:   [[TMP52:%.*]] = bitcast <8 x i64> [[TMP51]] to <16 x i32>
// CHECK:   [[TMP55:%.*]] = icmp ugt <16 x i32> [[TMP50]], [[TMP52]]
// CHECK:   [[TMP56:%.*]] = select <16 x i1> [[TMP55]], <16 x i32> [[TMP50]], <16 x i32> [[TMP52]]
// CHECK:   [[TMP57:%.*]] = bitcast <16 x i32> [[TMP56]] to <8 x i64>
// CHECK:   store <8 x i64> [[TMP57]], <8 x i64>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP58:%.*]] = load <8 x i64>, <8 x i64>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP59:%.*]] = bitcast <8 x i64> [[TMP58]] to <16 x i32>
// CHECK:   [[TMP60:%.*]] = load <8 x i64>, <8 x i64>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP61:%.*]] = bitcast <8 x i64> [[TMP60]] to <16 x i32>
// CHECK:   [[SHUFFLE8_I:%.*]] = shufflevector <16 x i32> [[TMP59]], <16 x i32> [[TMP61]], <16 x i32> <i32 0, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP62:%.*]] = bitcast <16 x i32> [[SHUFFLE8_I]] to <8 x i64>
// CHECK:   [[TMP63:%.*]] = load <8 x i64>, <8 x i64>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP64:%.*]] = bitcast <8 x i64> [[TMP63]] to <16 x i32>
// CHECK:   [[TMP65:%.*]] = load <8 x i64>, <8 x i64>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP66:%.*]] = bitcast <8 x i64> [[TMP65]] to <16 x i32>
// CHECK:   [[SHUFFLE9_I:%.*]] = shufflevector <16 x i32> [[TMP64]], <16 x i32> [[TMP66]], <16 x i32> <i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP67:%.*]] = bitcast <16 x i32> [[SHUFFLE9_I]] to <8 x i64>
// CHECK:   store <8 x i64> [[TMP62]], <8 x i64>* [[__A_ADDR_I12_I]], align 64
// CHECK:   store <8 x i64> [[TMP67]], <8 x i64>* [[__B_ADDR_I13_I]], align 64
// CHECK:   [[TMP68:%.*]] = load <8 x i64>, <8 x i64>* [[__A_ADDR_I12_I]], align 64
// CHECK:   [[TMP69:%.*]] = bitcast <8 x i64> [[TMP68]] to <16 x i32>
// CHECK:   [[TMP70:%.*]] = load <8 x i64>, <8 x i64>* [[__B_ADDR_I13_I]], align 64
// CHECK:   [[TMP71:%.*]] = bitcast <8 x i64> [[TMP70]] to <16 x i32>
// CHECK:   [[TMP74:%.*]] = icmp ugt <16 x i32> [[TMP69]], [[TMP71]]
// CHECK:   [[TMP75:%.*]] = select <16 x i1> [[TMP74]], <16 x i32> [[TMP69]], <16 x i32> [[TMP71]]
// CHECK:   [[TMP76:%.*]] = bitcast <16 x i32> [[TMP75]] to <8 x i64>
// CHECK:   store <8 x i64> [[TMP76]], <8 x i64>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP77:%.*]] = load <8 x i64>, <8 x i64>* [[A_ADDR_I]], align 64
// CHECK:   [[VECEXT_I:%.*]] = extractelement <8 x i64> [[TMP77]], i32 0
// CHECK:   [[CONV_I:%.*]] = trunc i64 [[VECEXT_I]] to i32
// CHECK:   ret i32 [[CONV_I]]
unsigned int test_mm512_reduce_max_epu32(__m512i __W){
  return _mm512_reduce_max_epu32(__W); 
}

// CHECK-LABEL: define float @test_mm512_reduce_max_ps(<16 x float> %__W) #0 {
// CHECK:   [[_COMPOUNDLITERAL_I_I17_I:%.*]] = alloca <16 x float>, align 64
// CHECK:   [[__A_ADDR_I18_I:%.*]] = alloca <16 x float>, align 64
// CHECK:   [[__B_ADDR_I19_I:%.*]] = alloca <16 x float>, align 64
// CHECK:   [[_COMPOUNDLITERAL_I_I14_I:%.*]] = alloca <16 x float>, align 64
// CHECK:   [[__A_ADDR_I15_I:%.*]] = alloca <16 x float>, align 64
// CHECK:   [[__B_ADDR_I16_I:%.*]] = alloca <16 x float>, align 64
// CHECK:   [[_COMPOUNDLITERAL_I_I11_I:%.*]] = alloca <16 x float>, align 64
// CHECK:   [[__A_ADDR_I12_I:%.*]] = alloca <16 x float>, align 64
// CHECK:   [[__B_ADDR_I13_I:%.*]] = alloca <16 x float>, align 64
// CHECK:   [[_COMPOUNDLITERAL_I_I_I:%.*]] = alloca <16 x float>, align 64
// CHECK:   [[__A_ADDR_I_I:%.*]] = alloca <16 x float>, align 64
// CHECK:   [[__B_ADDR_I_I:%.*]] = alloca <16 x float>, align 64
// CHECK:   [[A_ADDR_I:%.*]] = alloca <16 x float>, align 64
// CHECK:   [[__W_ADDR:%.*]] = alloca <16 x float>, align 64
// CHECK:   store <16 x float> %__W, <16 x float>* [[__W_ADDR]], align 64
// CHECK:   [[TMP0:%.*]] = load <16 x float>, <16 x float>* [[__W_ADDR]], align 64
// CHECK:   store <16 x float> [[TMP0]], <16 x float>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP1:%.*]] = load <16 x float>, <16 x float>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP2:%.*]] = load <16 x float>, <16 x float>* [[A_ADDR_I]], align 64
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <16 x float> [[TMP1]], <16 x float> [[TMP2]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP3:%.*]] = load <16 x float>, <16 x float>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP4:%.*]] = load <16 x float>, <16 x float>* [[A_ADDR_I]], align 64
// CHECK:   [[SHUFFLE1_I:%.*]] = shufflevector <16 x float> [[TMP3]], <16 x float> [[TMP4]], <16 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   store <16 x float> [[SHUFFLE_I]], <16 x float>* [[__A_ADDR_I_I]], align 64
// CHECK:   store <16 x float> [[SHUFFLE1_I]], <16 x float>* [[__B_ADDR_I_I]], align 64
// CHECK:   [[TMP5:%.*]] = load <16 x float>, <16 x float>* [[__A_ADDR_I_I]], align 64
// CHECK:   [[TMP6:%.*]] = load <16 x float>, <16 x float>* [[__B_ADDR_I_I]], align 64
// CHECK:   store <16 x float> zeroinitializer, <16 x float>* [[_COMPOUNDLITERAL_I_I_I]], align 64
// CHECK:   [[TMP7:%.*]] = load <16 x float>, <16 x float>* [[_COMPOUNDLITERAL_I_I_I]], align 64
// CHECK:   [[TMP8:%.*]] = call <16 x float> @llvm.x86.avx512.mask.max.ps.512(<16 x float> [[TMP5]], <16 x float> [[TMP6]], <16 x float> [[TMP7]], i16 -1, i32 4) #2
// CHECK:   store <16 x float> [[TMP8]], <16 x float>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP9:%.*]] = load <16 x float>, <16 x float>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP10:%.*]] = load <16 x float>, <16 x float>* [[A_ADDR_I]], align 64
// CHECK:   [[SHUFFLE2_I:%.*]] = shufflevector <16 x float> [[TMP9]], <16 x float> [[TMP10]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP11:%.*]] = load <16 x float>, <16 x float>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP12:%.*]] = load <16 x float>, <16 x float>* [[A_ADDR_I]], align 64
// CHECK:   [[SHUFFLE3_I:%.*]] = shufflevector <16 x float> [[TMP11]], <16 x float> [[TMP12]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   store <16 x float> [[SHUFFLE2_I]], <16 x float>* [[__A_ADDR_I18_I]], align 64
// CHECK:   store <16 x float> [[SHUFFLE3_I]], <16 x float>* [[__B_ADDR_I19_I]], align 64
// CHECK:   [[TMP13:%.*]] = load <16 x float>, <16 x float>* [[__A_ADDR_I18_I]], align 64
// CHECK:   [[TMP14:%.*]] = load <16 x float>, <16 x float>* [[__B_ADDR_I19_I]], align 64
// CHECK:   store <16 x float> zeroinitializer, <16 x float>* [[_COMPOUNDLITERAL_I_I17_I]], align 64
// CHECK:   [[TMP15:%.*]] = load <16 x float>, <16 x float>* [[_COMPOUNDLITERAL_I_I17_I]], align 64
// CHECK:   [[TMP16:%.*]] = call <16 x float> @llvm.x86.avx512.mask.max.ps.512(<16 x float> [[TMP13]], <16 x float> [[TMP14]], <16 x float> [[TMP15]], i16 -1, i32 4) #2
// CHECK:   store <16 x float> [[TMP16]], <16 x float>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP17:%.*]] = load <16 x float>, <16 x float>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP18:%.*]] = load <16 x float>, <16 x float>* [[A_ADDR_I]], align 64
// CHECK:   [[SHUFFLE5_I:%.*]] = shufflevector <16 x float> [[TMP17]], <16 x float> [[TMP18]], <16 x i32> <i32 0, i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP19:%.*]] = load <16 x float>, <16 x float>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP20:%.*]] = load <16 x float>, <16 x float>* [[A_ADDR_I]], align 64
// CHECK:   [[SHUFFLE6_I:%.*]] = shufflevector <16 x float> [[TMP19]], <16 x float> [[TMP20]], <16 x i32> <i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   store <16 x float> [[SHUFFLE5_I]], <16 x float>* [[__A_ADDR_I15_I]], align 64
// CHECK:   store <16 x float> [[SHUFFLE6_I]], <16 x float>* [[__B_ADDR_I16_I]], align 64
// CHECK:   [[TMP21:%.*]] = load <16 x float>, <16 x float>* [[__A_ADDR_I15_I]], align 64
// CHECK:   [[TMP22:%.*]] = load <16 x float>, <16 x float>* [[__B_ADDR_I16_I]], align 64
// CHECK:   store <16 x float> zeroinitializer, <16 x float>* [[_COMPOUNDLITERAL_I_I14_I]], align 64
// CHECK:   [[TMP23:%.*]] = load <16 x float>, <16 x float>* [[_COMPOUNDLITERAL_I_I14_I]], align 64
// CHECK:   [[TMP24:%.*]] = call <16 x float> @llvm.x86.avx512.mask.max.ps.512(<16 x float> [[TMP21]], <16 x float> [[TMP22]], <16 x float> [[TMP23]], i16 -1, i32 4) #2
// CHECK:   store <16 x float> [[TMP24]], <16 x float>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP25:%.*]] = load <16 x float>, <16 x float>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP26:%.*]] = load <16 x float>, <16 x float>* [[A_ADDR_I]], align 64
// CHECK:   [[SHUFFLE8_I:%.*]] = shufflevector <16 x float> [[TMP25]], <16 x float> [[TMP26]], <16 x i32> <i32 0, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP27:%.*]] = load <16 x float>, <16 x float>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP28:%.*]] = load <16 x float>, <16 x float>* [[A_ADDR_I]], align 64
// CHECK:   [[SHUFFLE9_I:%.*]] = shufflevector <16 x float> [[TMP27]], <16 x float> [[TMP28]], <16 x i32> <i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   store <16 x float> [[SHUFFLE8_I]], <16 x float>* [[__A_ADDR_I12_I]], align 64
// CHECK:   store <16 x float> [[SHUFFLE9_I]], <16 x float>* [[__B_ADDR_I13_I]], align 64
// CHECK:   [[TMP29:%.*]] = load <16 x float>, <16 x float>* [[__A_ADDR_I12_I]], align 64
// CHECK:   [[TMP30:%.*]] = load <16 x float>, <16 x float>* [[__B_ADDR_I13_I]], align 64
// CHECK:   store <16 x float> zeroinitializer, <16 x float>* [[_COMPOUNDLITERAL_I_I11_I]], align 64
// CHECK:   [[TMP31:%.*]] = load <16 x float>, <16 x float>* [[_COMPOUNDLITERAL_I_I11_I]], align 64
// CHECK:   [[TMP32:%.*]] = call <16 x float> @llvm.x86.avx512.mask.max.ps.512(<16 x float> [[TMP29]], <16 x float> [[TMP30]], <16 x float> [[TMP31]], i16 -1, i32 4) #2
// CHECK:   store <16 x float> [[TMP32]], <16 x float>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP33:%.*]] = load <16 x float>, <16 x float>* [[A_ADDR_I]], align 64
// CHECK:   [[VECEXT_I:%.*]] = extractelement <16 x float> [[TMP33]], i32 0
// CHECK:   ret float [[VECEXT_I]]
float test_mm512_reduce_max_ps(__m512 __W){
  return _mm512_reduce_max_ps(__W); 
}

// CHECK-LABEL: define i32 @test_mm512_reduce_min_epi32(<8 x i64> %__W) #0 {
// CHECK:   [[__A_ADDR_I18_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__B_ADDR_I19_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__A_ADDR_I15_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__B_ADDR_I16_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__A_ADDR_I12_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__B_ADDR_I13_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__A_ADDR_I_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__B_ADDR_I_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[A_ADDR_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__W_ADDR:%.*]] = alloca <8 x i64>, align 64
// CHECK:   store <8 x i64> %__W, <8 x i64>* [[__W_ADDR]], align 64
// CHECK:   [[TMP0:%.*]] = load <8 x i64>, <8 x i64>* [[__W_ADDR]], align 64
// CHECK:   store <8 x i64> [[TMP0]], <8 x i64>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP1:%.*]] = load <8 x i64>, <8 x i64>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP2:%.*]] = bitcast <8 x i64> [[TMP1]] to <16 x i32>
// CHECK:   [[TMP3:%.*]] = load <8 x i64>, <8 x i64>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP4:%.*]] = bitcast <8 x i64> [[TMP3]] to <16 x i32>
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <16 x i32> [[TMP2]], <16 x i32> [[TMP4]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP5:%.*]] = bitcast <16 x i32> [[SHUFFLE_I]] to <8 x i64>
// CHECK:   [[TMP6:%.*]] = load <8 x i64>, <8 x i64>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP7:%.*]] = bitcast <8 x i64> [[TMP6]] to <16 x i32>
// CHECK:   [[TMP8:%.*]] = load <8 x i64>, <8 x i64>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP9:%.*]] = bitcast <8 x i64> [[TMP8]] to <16 x i32>
// CHECK:   [[SHUFFLE1_I:%.*]] = shufflevector <16 x i32> [[TMP7]], <16 x i32> [[TMP9]], <16 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP10:%.*]] = bitcast <16 x i32> [[SHUFFLE1_I]] to <8 x i64>
// CHECK:   store <8 x i64> [[TMP5]], <8 x i64>* [[__A_ADDR_I_I]], align 64
// CHECK:   store <8 x i64> [[TMP10]], <8 x i64>* [[__B_ADDR_I_I]], align 64
// CHECK:   [[TMP11:%.*]] = load <8 x i64>, <8 x i64>* [[__A_ADDR_I_I]], align 64
// CHECK:   [[TMP12:%.*]] = bitcast <8 x i64> [[TMP11]] to <16 x i32>
// CHECK:   [[TMP13:%.*]] = load <8 x i64>, <8 x i64>* [[__B_ADDR_I_I]], align 64
// CHECK:   [[TMP14:%.*]] = bitcast <8 x i64> [[TMP13]] to <16 x i32>
// CHECK:   [[TMP17:%.*]] = icmp slt <16 x i32> [[TMP12]], [[TMP14]]
// CHECK:   [[TMP18:%.*]] = select <16 x i1> [[TMP17]], <16 x i32> [[TMP12]], <16 x i32> [[TMP14]]
// CHECK:   [[TMP19:%.*]] = bitcast <16 x i32> [[TMP18]] to <8 x i64>
// CHECK:   store <8 x i64> [[TMP19]], <8 x i64>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP20:%.*]] = load <8 x i64>, <8 x i64>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP21:%.*]] = bitcast <8 x i64> [[TMP20]] to <16 x i32>
// CHECK:   [[TMP22:%.*]] = load <8 x i64>, <8 x i64>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP23:%.*]] = bitcast <8 x i64> [[TMP22]] to <16 x i32>
// CHECK:   [[SHUFFLE2_I:%.*]] = shufflevector <16 x i32> [[TMP21]], <16 x i32> [[TMP23]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP24:%.*]] = bitcast <16 x i32> [[SHUFFLE2_I]] to <8 x i64>
// CHECK:   [[TMP25:%.*]] = load <8 x i64>, <8 x i64>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP26:%.*]] = bitcast <8 x i64> [[TMP25]] to <16 x i32>
// CHECK:   [[TMP27:%.*]] = load <8 x i64>, <8 x i64>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP28:%.*]] = bitcast <8 x i64> [[TMP27]] to <16 x i32>
// CHECK:   [[SHUFFLE3_I:%.*]] = shufflevector <16 x i32> [[TMP26]], <16 x i32> [[TMP28]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP29:%.*]] = bitcast <16 x i32> [[SHUFFLE3_I]] to <8 x i64>
// CHECK:   store <8 x i64> [[TMP24]], <8 x i64>* [[__A_ADDR_I18_I]], align 64
// CHECK:   store <8 x i64> [[TMP29]], <8 x i64>* [[__B_ADDR_I19_I]], align 64
// CHECK:   [[TMP30:%.*]] = load <8 x i64>, <8 x i64>* [[__A_ADDR_I18_I]], align 64
// CHECK:   [[TMP31:%.*]] = bitcast <8 x i64> [[TMP30]] to <16 x i32>
// CHECK:   [[TMP32:%.*]] = load <8 x i64>, <8 x i64>* [[__B_ADDR_I19_I]], align 64
// CHECK:   [[TMP33:%.*]] = bitcast <8 x i64> [[TMP32]] to <16 x i32>
// CHECK:   [[TMP36:%.*]] = icmp slt <16 x i32> [[TMP31]], [[TMP33]]
// CHECK:   [[TMP37:%.*]] = select <16 x i1> [[TMP36]], <16 x i32> [[TMP31]], <16 x i32> [[TMP33]]
// CHECK:   [[TMP38:%.*]] = bitcast <16 x i32> [[TMP37]] to <8 x i64>
// CHECK:   store <8 x i64> [[TMP38]], <8 x i64>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP39:%.*]] = load <8 x i64>, <8 x i64>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP40:%.*]] = bitcast <8 x i64> [[TMP39]] to <16 x i32>
// CHECK:   [[TMP41:%.*]] = load <8 x i64>, <8 x i64>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP42:%.*]] = bitcast <8 x i64> [[TMP41]] to <16 x i32>
// CHECK:   [[SHUFFLE5_I:%.*]] = shufflevector <16 x i32> [[TMP40]], <16 x i32> [[TMP42]], <16 x i32> <i32 0, i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP43:%.*]] = bitcast <16 x i32> [[SHUFFLE5_I]] to <8 x i64>
// CHECK:   [[TMP44:%.*]] = load <8 x i64>, <8 x i64>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP45:%.*]] = bitcast <8 x i64> [[TMP44]] to <16 x i32>
// CHECK:   [[TMP46:%.*]] = load <8 x i64>, <8 x i64>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP47:%.*]] = bitcast <8 x i64> [[TMP46]] to <16 x i32>
// CHECK:   [[SHUFFLE6_I:%.*]] = shufflevector <16 x i32> [[TMP45]], <16 x i32> [[TMP47]], <16 x i32> <i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP48:%.*]] = bitcast <16 x i32> [[SHUFFLE6_I]] to <8 x i64>
// CHECK:   store <8 x i64> [[TMP43]], <8 x i64>* [[__A_ADDR_I15_I]], align 64
// CHECK:   store <8 x i64> [[TMP48]], <8 x i64>* [[__B_ADDR_I16_I]], align 64
// CHECK:   [[TMP49:%.*]] = load <8 x i64>, <8 x i64>* [[__A_ADDR_I15_I]], align 64
// CHECK:   [[TMP50:%.*]] = bitcast <8 x i64> [[TMP49]] to <16 x i32>
// CHECK:   [[TMP51:%.*]] = load <8 x i64>, <8 x i64>* [[__B_ADDR_I16_I]], align 64
// CHECK:   [[TMP52:%.*]] = bitcast <8 x i64> [[TMP51]] to <16 x i32>
// CHECK:   [[TMP55:%.*]] = icmp slt <16 x i32> [[TMP50]], [[TMP52]]
// CHECK:   [[TMP56:%.*]] = select <16 x i1> [[TMP55]], <16 x i32> [[TMP50]], <16 x i32> [[TMP52]]
// CHECK:   [[TMP57:%.*]] = bitcast <16 x i32> [[TMP56]] to <8 x i64>
// CHECK:   store <8 x i64> [[TMP57]], <8 x i64>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP58:%.*]] = load <8 x i64>, <8 x i64>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP59:%.*]] = bitcast <8 x i64> [[TMP58]] to <16 x i32>
// CHECK:   [[TMP60:%.*]] = load <8 x i64>, <8 x i64>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP61:%.*]] = bitcast <8 x i64> [[TMP60]] to <16 x i32>
// CHECK:   [[SHUFFLE8_I:%.*]] = shufflevector <16 x i32> [[TMP59]], <16 x i32> [[TMP61]], <16 x i32> <i32 0, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP62:%.*]] = bitcast <16 x i32> [[SHUFFLE8_I]] to <8 x i64>
// CHECK:   [[TMP63:%.*]] = load <8 x i64>, <8 x i64>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP64:%.*]] = bitcast <8 x i64> [[TMP63]] to <16 x i32>
// CHECK:   [[TMP65:%.*]] = load <8 x i64>, <8 x i64>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP66:%.*]] = bitcast <8 x i64> [[TMP65]] to <16 x i32>
// CHECK:   [[SHUFFLE9_I:%.*]] = shufflevector <16 x i32> [[TMP64]], <16 x i32> [[TMP66]], <16 x i32> <i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP67:%.*]] = bitcast <16 x i32> [[SHUFFLE9_I]] to <8 x i64>
// CHECK:   store <8 x i64> [[TMP62]], <8 x i64>* [[__A_ADDR_I12_I]], align 64
// CHECK:   store <8 x i64> [[TMP67]], <8 x i64>* [[__B_ADDR_I13_I]], align 64
// CHECK:   [[TMP68:%.*]] = load <8 x i64>, <8 x i64>* [[__A_ADDR_I12_I]], align 64
// CHECK:   [[TMP69:%.*]] = bitcast <8 x i64> [[TMP68]] to <16 x i32>
// CHECK:   [[TMP70:%.*]] = load <8 x i64>, <8 x i64>* [[__B_ADDR_I13_I]], align 64
// CHECK:   [[TMP71:%.*]] = bitcast <8 x i64> [[TMP70]] to <16 x i32>
// CHECK:   [[TMP74:%.*]] = icmp slt <16 x i32> [[TMP69]], [[TMP71]]
// CHECK:   [[TMP75:%.*]] = select <16 x i1> [[TMP74]], <16 x i32> [[TMP69]], <16 x i32> [[TMP71]]
// CHECK:   [[TMP76:%.*]] = bitcast <16 x i32> [[TMP75]] to <8 x i64>
// CHECK:   store <8 x i64> [[TMP76]], <8 x i64>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP77:%.*]] = load <8 x i64>, <8 x i64>* [[A_ADDR_I]], align 64
// CHECK:   [[VECEXT_I:%.*]] = extractelement <8 x i64> [[TMP77]], i32 0
// CHECK:   [[CONV_I:%.*]] = trunc i64 [[VECEXT_I]] to i32
// CHECK:   ret i32 [[CONV_I]]
int test_mm512_reduce_min_epi32(__m512i __W){
  return _mm512_reduce_min_epi32(__W);
}

// CHECK-LABEL: define i32 @test_mm512_reduce_min_epu32(<8 x i64> %__W) #0 {
// CHECK:   [[__A_ADDR_I18_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__B_ADDR_I19_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__A_ADDR_I15_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__B_ADDR_I16_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__A_ADDR_I12_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__B_ADDR_I13_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__A_ADDR_I_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__B_ADDR_I_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[A_ADDR_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__W_ADDR:%.*]] = alloca <8 x i64>, align 64
// CHECK:   store <8 x i64> %__W, <8 x i64>* [[__W_ADDR]], align 64
// CHECK:   [[TMP0:%.*]] = load <8 x i64>, <8 x i64>* [[__W_ADDR]], align 64
// CHECK:   store <8 x i64> [[TMP0]], <8 x i64>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP1:%.*]] = load <8 x i64>, <8 x i64>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP2:%.*]] = bitcast <8 x i64> [[TMP1]] to <16 x i32>
// CHECK:   [[TMP3:%.*]] = load <8 x i64>, <8 x i64>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP4:%.*]] = bitcast <8 x i64> [[TMP3]] to <16 x i32>
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <16 x i32> [[TMP2]], <16 x i32> [[TMP4]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP5:%.*]] = bitcast <16 x i32> [[SHUFFLE_I]] to <8 x i64>
// CHECK:   [[TMP6:%.*]] = load <8 x i64>, <8 x i64>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP7:%.*]] = bitcast <8 x i64> [[TMP6]] to <16 x i32>
// CHECK:   [[TMP8:%.*]] = load <8 x i64>, <8 x i64>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP9:%.*]] = bitcast <8 x i64> [[TMP8]] to <16 x i32>
// CHECK:   [[SHUFFLE1_I:%.*]] = shufflevector <16 x i32> [[TMP7]], <16 x i32> [[TMP9]], <16 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP10:%.*]] = bitcast <16 x i32> [[SHUFFLE1_I]] to <8 x i64>
// CHECK:   store <8 x i64> [[TMP5]], <8 x i64>* [[__A_ADDR_I_I]], align 64
// CHECK:   store <8 x i64> [[TMP10]], <8 x i64>* [[__B_ADDR_I_I]], align 64
// CHECK:   [[TMP11:%.*]] = load <8 x i64>, <8 x i64>* [[__A_ADDR_I_I]], align 64
// CHECK:   [[TMP12:%.*]] = bitcast <8 x i64> [[TMP11]] to <16 x i32>
// CHECK:   [[TMP13:%.*]] = load <8 x i64>, <8 x i64>* [[__B_ADDR_I_I]], align 64
// CHECK:   [[TMP14:%.*]] = bitcast <8 x i64> [[TMP13]] to <16 x i32>
// CHECK:   [[TMP17:%.*]] = icmp ult <16 x i32> [[TMP12]], [[TMP14]]
// CHECK:   [[TMP18:%.*]] = select <16 x i1> [[TMP17]], <16 x i32> [[TMP12]], <16 x i32> [[TMP14]]
// CHECK:   [[TMP19:%.*]] = bitcast <16 x i32> [[TMP18]] to <8 x i64>
// CHECK:   store <8 x i64> [[TMP19]], <8 x i64>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP20:%.*]] = load <8 x i64>, <8 x i64>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP21:%.*]] = bitcast <8 x i64> [[TMP20]] to <16 x i32>
// CHECK:   [[TMP22:%.*]] = load <8 x i64>, <8 x i64>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP23:%.*]] = bitcast <8 x i64> [[TMP22]] to <16 x i32>
// CHECK:   [[SHUFFLE2_I:%.*]] = shufflevector <16 x i32> [[TMP21]], <16 x i32> [[TMP23]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP24:%.*]] = bitcast <16 x i32> [[SHUFFLE2_I]] to <8 x i64>
// CHECK:   [[TMP25:%.*]] = load <8 x i64>, <8 x i64>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP26:%.*]] = bitcast <8 x i64> [[TMP25]] to <16 x i32>
// CHECK:   [[TMP27:%.*]] = load <8 x i64>, <8 x i64>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP28:%.*]] = bitcast <8 x i64> [[TMP27]] to <16 x i32>
// CHECK:   [[SHUFFLE3_I:%.*]] = shufflevector <16 x i32> [[TMP26]], <16 x i32> [[TMP28]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP29:%.*]] = bitcast <16 x i32> [[SHUFFLE3_I]] to <8 x i64>
// CHECK:   store <8 x i64> [[TMP24]], <8 x i64>* [[__A_ADDR_I18_I]], align 64
// CHECK:   store <8 x i64> [[TMP29]], <8 x i64>* [[__B_ADDR_I19_I]], align 64
// CHECK:   [[TMP30:%.*]] = load <8 x i64>, <8 x i64>* [[__A_ADDR_I18_I]], align 64
// CHECK:   [[TMP31:%.*]] = bitcast <8 x i64> [[TMP30]] to <16 x i32>
// CHECK:   [[TMP32:%.*]] = load <8 x i64>, <8 x i64>* [[__B_ADDR_I19_I]], align 64
// CHECK:   [[TMP33:%.*]] = bitcast <8 x i64> [[TMP32]] to <16 x i32>
// CHECK:   [[TMP36:%.*]] = icmp ult <16 x i32> [[TMP31]], [[TMP33]]
// CHECK:   [[TMP37:%.*]] = select <16 x i1> [[TMP36]], <16 x i32> [[TMP31]], <16 x i32> [[TMP33]]
// CHECK:   [[TMP38:%.*]] = bitcast <16 x i32> [[TMP37]] to <8 x i64>
// CHECK:   store <8 x i64> [[TMP38]], <8 x i64>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP39:%.*]] = load <8 x i64>, <8 x i64>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP40:%.*]] = bitcast <8 x i64> [[TMP39]] to <16 x i32>
// CHECK:   [[TMP41:%.*]] = load <8 x i64>, <8 x i64>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP42:%.*]] = bitcast <8 x i64> [[TMP41]] to <16 x i32>
// CHECK:   [[SHUFFLE5_I:%.*]] = shufflevector <16 x i32> [[TMP40]], <16 x i32> [[TMP42]], <16 x i32> <i32 0, i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP43:%.*]] = bitcast <16 x i32> [[SHUFFLE5_I]] to <8 x i64>
// CHECK:   [[TMP44:%.*]] = load <8 x i64>, <8 x i64>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP45:%.*]] = bitcast <8 x i64> [[TMP44]] to <16 x i32>
// CHECK:   [[TMP46:%.*]] = load <8 x i64>, <8 x i64>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP47:%.*]] = bitcast <8 x i64> [[TMP46]] to <16 x i32>
// CHECK:   [[SHUFFLE6_I:%.*]] = shufflevector <16 x i32> [[TMP45]], <16 x i32> [[TMP47]], <16 x i32> <i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP48:%.*]] = bitcast <16 x i32> [[SHUFFLE6_I]] to <8 x i64>
// CHECK:   store <8 x i64> [[TMP43]], <8 x i64>* [[__A_ADDR_I15_I]], align 64
// CHECK:   store <8 x i64> [[TMP48]], <8 x i64>* [[__B_ADDR_I16_I]], align 64
// CHECK:   [[TMP49:%.*]] = load <8 x i64>, <8 x i64>* [[__A_ADDR_I15_I]], align 64
// CHECK:   [[TMP50:%.*]] = bitcast <8 x i64> [[TMP49]] to <16 x i32>
// CHECK:   [[TMP51:%.*]] = load <8 x i64>, <8 x i64>* [[__B_ADDR_I16_I]], align 64
// CHECK:   [[TMP52:%.*]] = bitcast <8 x i64> [[TMP51]] to <16 x i32>
// CHECK:   [[TMP55:%.*]] = icmp ult <16 x i32> [[TMP50]], [[TMP52]]
// CHECK:   [[TMP56:%.*]] = select <16 x i1> [[TMP55]], <16 x i32> [[TMP50]], <16 x i32> [[TMP52]]
// CHECK:   [[TMP57:%.*]] = bitcast <16 x i32> [[TMP56]] to <8 x i64>
// CHECK:   store <8 x i64> [[TMP57]], <8 x i64>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP58:%.*]] = load <8 x i64>, <8 x i64>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP59:%.*]] = bitcast <8 x i64> [[TMP58]] to <16 x i32>
// CHECK:   [[TMP60:%.*]] = load <8 x i64>, <8 x i64>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP61:%.*]] = bitcast <8 x i64> [[TMP60]] to <16 x i32>
// CHECK:   [[SHUFFLE8_I:%.*]] = shufflevector <16 x i32> [[TMP59]], <16 x i32> [[TMP61]], <16 x i32> <i32 0, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP62:%.*]] = bitcast <16 x i32> [[SHUFFLE8_I]] to <8 x i64>
// CHECK:   [[TMP63:%.*]] = load <8 x i64>, <8 x i64>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP64:%.*]] = bitcast <8 x i64> [[TMP63]] to <16 x i32>
// CHECK:   [[TMP65:%.*]] = load <8 x i64>, <8 x i64>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP66:%.*]] = bitcast <8 x i64> [[TMP65]] to <16 x i32>
// CHECK:   [[SHUFFLE9_I:%.*]] = shufflevector <16 x i32> [[TMP64]], <16 x i32> [[TMP66]], <16 x i32> <i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP67:%.*]] = bitcast <16 x i32> [[SHUFFLE9_I]] to <8 x i64>
// CHECK:   store <8 x i64> [[TMP62]], <8 x i64>* [[__A_ADDR_I12_I]], align 64
// CHECK:   store <8 x i64> [[TMP67]], <8 x i64>* [[__B_ADDR_I13_I]], align 64
// CHECK:   [[TMP68:%.*]] = load <8 x i64>, <8 x i64>* [[__A_ADDR_I12_I]], align 64
// CHECK:   [[TMP69:%.*]] = bitcast <8 x i64> [[TMP68]] to <16 x i32>
// CHECK:   [[TMP70:%.*]] = load <8 x i64>, <8 x i64>* [[__B_ADDR_I13_I]], align 64
// CHECK:   [[TMP71:%.*]] = bitcast <8 x i64> [[TMP70]] to <16 x i32>
// CHECK:   [[TMP74:%.*]] = icmp ult <16 x i32> [[TMP69]], [[TMP71]]
// CHECK:   [[TMP75:%.*]] = select <16 x i1> [[TMP74]], <16 x i32> [[TMP69]], <16 x i32> [[TMP71]]
// CHECK:   [[TMP76:%.*]] = bitcast <16 x i32> [[TMP75]] to <8 x i64>
// CHECK:   store <8 x i64> [[TMP76]], <8 x i64>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP77:%.*]] = load <8 x i64>, <8 x i64>* [[A_ADDR_I]], align 64
// CHECK:   [[VECEXT_I:%.*]] = extractelement <8 x i64> [[TMP77]], i32 0
// CHECK:   [[CONV_I:%.*]] = trunc i64 [[VECEXT_I]] to i32
// CHECK:   ret i32 [[CONV_I]]
unsigned int test_mm512_reduce_min_epu32(__m512i __W){
  return _mm512_reduce_min_epu32(__W); 
}

// CHECK-LABEL: define float @test_mm512_reduce_min_ps(<16 x float> %__W) #0 {
// CHECK:   [[_COMPOUNDLITERAL_I_I17_I:%.*]] = alloca <16 x float>, align 64
// CHECK:   [[__A_ADDR_I18_I:%.*]] = alloca <16 x float>, align 64
// CHECK:   [[__B_ADDR_I19_I:%.*]] = alloca <16 x float>, align 64
// CHECK:   [[_COMPOUNDLITERAL_I_I14_I:%.*]] = alloca <16 x float>, align 64
// CHECK:   [[__A_ADDR_I15_I:%.*]] = alloca <16 x float>, align 64
// CHECK:   [[__B_ADDR_I16_I:%.*]] = alloca <16 x float>, align 64
// CHECK:   [[_COMPOUNDLITERAL_I_I11_I:%.*]] = alloca <16 x float>, align 64
// CHECK:   [[__A_ADDR_I12_I:%.*]] = alloca <16 x float>, align 64
// CHECK:   [[__B_ADDR_I13_I:%.*]] = alloca <16 x float>, align 64
// CHECK:   [[_COMPOUNDLITERAL_I_I_I:%.*]] = alloca <16 x float>, align 64
// CHECK:   [[__A_ADDR_I_I:%.*]] = alloca <16 x float>, align 64
// CHECK:   [[__B_ADDR_I_I:%.*]] = alloca <16 x float>, align 64
// CHECK:   [[A_ADDR_I:%.*]] = alloca <16 x float>, align 64
// CHECK:   [[__W_ADDR:%.*]] = alloca <16 x float>, align 64
// CHECK:   store <16 x float> %__W, <16 x float>* [[__W_ADDR]], align 64
// CHECK:   [[TMP0:%.*]] = load <16 x float>, <16 x float>* [[__W_ADDR]], align 64
// CHECK:   store <16 x float> [[TMP0]], <16 x float>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP1:%.*]] = load <16 x float>, <16 x float>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP2:%.*]] = load <16 x float>, <16 x float>* [[A_ADDR_I]], align 64
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <16 x float> [[TMP1]], <16 x float> [[TMP2]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP3:%.*]] = load <16 x float>, <16 x float>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP4:%.*]] = load <16 x float>, <16 x float>* [[A_ADDR_I]], align 64
// CHECK:   [[SHUFFLE1_I:%.*]] = shufflevector <16 x float> [[TMP3]], <16 x float> [[TMP4]], <16 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   store <16 x float> [[SHUFFLE_I]], <16 x float>* [[__A_ADDR_I_I]], align 64
// CHECK:   store <16 x float> [[SHUFFLE1_I]], <16 x float>* [[__B_ADDR_I_I]], align 64
// CHECK:   [[TMP5:%.*]] = load <16 x float>, <16 x float>* [[__A_ADDR_I_I]], align 64
// CHECK:   [[TMP6:%.*]] = load <16 x float>, <16 x float>* [[__B_ADDR_I_I]], align 64
// CHECK:   store <16 x float> zeroinitializer, <16 x float>* [[_COMPOUNDLITERAL_I_I_I]], align 64
// CHECK:   [[TMP7:%.*]] = load <16 x float>, <16 x float>* [[_COMPOUNDLITERAL_I_I_I]], align 64
// CHECK:   [[TMP8:%.*]] = call <16 x float> @llvm.x86.avx512.mask.min.ps.512(<16 x float> [[TMP5]], <16 x float> [[TMP6]], <16 x float> [[TMP7]], i16 -1, i32 4) #2
// CHECK:   store <16 x float> [[TMP8]], <16 x float>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP9:%.*]] = load <16 x float>, <16 x float>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP10:%.*]] = load <16 x float>, <16 x float>* [[A_ADDR_I]], align 64
// CHECK:   [[SHUFFLE2_I:%.*]] = shufflevector <16 x float> [[TMP9]], <16 x float> [[TMP10]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP11:%.*]] = load <16 x float>, <16 x float>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP12:%.*]] = load <16 x float>, <16 x float>* [[A_ADDR_I]], align 64
// CHECK:   [[SHUFFLE3_I:%.*]] = shufflevector <16 x float> [[TMP11]], <16 x float> [[TMP12]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   store <16 x float> [[SHUFFLE2_I]], <16 x float>* [[__A_ADDR_I18_I]], align 64
// CHECK:   store <16 x float> [[SHUFFLE3_I]], <16 x float>* [[__B_ADDR_I19_I]], align 64
// CHECK:   [[TMP13:%.*]] = load <16 x float>, <16 x float>* [[__A_ADDR_I18_I]], align 64
// CHECK:   [[TMP14:%.*]] = load <16 x float>, <16 x float>* [[__B_ADDR_I19_I]], align 64
// CHECK:   store <16 x float> zeroinitializer, <16 x float>* [[_COMPOUNDLITERAL_I_I17_I]], align 64
// CHECK:   [[TMP15:%.*]] = load <16 x float>, <16 x float>* [[_COMPOUNDLITERAL_I_I17_I]], align 64
// CHECK:   [[TMP16:%.*]] = call <16 x float> @llvm.x86.avx512.mask.min.ps.512(<16 x float> [[TMP13]], <16 x float> [[TMP14]], <16 x float> [[TMP15]], i16 -1, i32 4) #2
// CHECK:   store <16 x float> [[TMP16]], <16 x float>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP17:%.*]] = load <16 x float>, <16 x float>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP18:%.*]] = load <16 x float>, <16 x float>* [[A_ADDR_I]], align 64
// CHECK:   [[SHUFFLE5_I:%.*]] = shufflevector <16 x float> [[TMP17]], <16 x float> [[TMP18]], <16 x i32> <i32 0, i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP19:%.*]] = load <16 x float>, <16 x float>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP20:%.*]] = load <16 x float>, <16 x float>* [[A_ADDR_I]], align 64
// CHECK:   [[SHUFFLE6_I:%.*]] = shufflevector <16 x float> [[TMP19]], <16 x float> [[TMP20]], <16 x i32> <i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   store <16 x float> [[SHUFFLE5_I]], <16 x float>* [[__A_ADDR_I15_I]], align 64
// CHECK:   store <16 x float> [[SHUFFLE6_I]], <16 x float>* [[__B_ADDR_I16_I]], align 64
// CHECK:   [[TMP21:%.*]] = load <16 x float>, <16 x float>* [[__A_ADDR_I15_I]], align 64
// CHECK:   [[TMP22:%.*]] = load <16 x float>, <16 x float>* [[__B_ADDR_I16_I]], align 64
// CHECK:   store <16 x float> zeroinitializer, <16 x float>* [[_COMPOUNDLITERAL_I_I14_I]], align 64
// CHECK:   [[TMP23:%.*]] = load <16 x float>, <16 x float>* [[_COMPOUNDLITERAL_I_I14_I]], align 64
// CHECK:   [[TMP24:%.*]] = call <16 x float> @llvm.x86.avx512.mask.min.ps.512(<16 x float> [[TMP21]], <16 x float> [[TMP22]], <16 x float> [[TMP23]], i16 -1, i32 4) #2
// CHECK:   store <16 x float> [[TMP24]], <16 x float>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP25:%.*]] = load <16 x float>, <16 x float>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP26:%.*]] = load <16 x float>, <16 x float>* [[A_ADDR_I]], align 64
// CHECK:   [[SHUFFLE8_I:%.*]] = shufflevector <16 x float> [[TMP25]], <16 x float> [[TMP26]], <16 x i32> <i32 0, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP27:%.*]] = load <16 x float>, <16 x float>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP28:%.*]] = load <16 x float>, <16 x float>* [[A_ADDR_I]], align 64
// CHECK:   [[SHUFFLE9_I:%.*]] = shufflevector <16 x float> [[TMP27]], <16 x float> [[TMP28]], <16 x i32> <i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   store <16 x float> [[SHUFFLE8_I]], <16 x float>* [[__A_ADDR_I12_I]], align 64
// CHECK:   store <16 x float> [[SHUFFLE9_I]], <16 x float>* [[__B_ADDR_I13_I]], align 64
// CHECK:   [[TMP29:%.*]] = load <16 x float>, <16 x float>* [[__A_ADDR_I12_I]], align 64
// CHECK:   [[TMP30:%.*]] = load <16 x float>, <16 x float>* [[__B_ADDR_I13_I]], align 64
// CHECK:   store <16 x float> zeroinitializer, <16 x float>* [[_COMPOUNDLITERAL_I_I11_I]], align 64
// CHECK:   [[TMP31:%.*]] = load <16 x float>, <16 x float>* [[_COMPOUNDLITERAL_I_I11_I]], align 64
// CHECK:   [[TMP32:%.*]] = call <16 x float> @llvm.x86.avx512.mask.min.ps.512(<16 x float> [[TMP29]], <16 x float> [[TMP30]], <16 x float> [[TMP31]], i16 -1, i32 4) #2
// CHECK:   store <16 x float> [[TMP32]], <16 x float>* [[A_ADDR_I]], align 64
// CHECK:   [[TMP33:%.*]] = load <16 x float>, <16 x float>* [[A_ADDR_I]], align 64
// CHECK:   [[VECEXT_I:%.*]] = extractelement <16 x float> [[TMP33]], i32 0
// CHECK:   ret float [[VECEXT_I]]
float test_mm512_reduce_min_ps(__m512 __W){
  return _mm512_reduce_min_ps(__W); 
}

// CHECK-LABEL: define i32 @test_mm512_mask_reduce_max_epi32(i16 zeroext %__M, <8 x i64> %__W) #0 {
// CHECK:   [[__A_ADDR_I19_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__B_ADDR_I20_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__A_ADDR_I16_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__B_ADDR_I17_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__A_ADDR_I13_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__B_ADDR_I14_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__A_ADDR_I_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__B_ADDR_I_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__S_ADDR_I_I:%.*]] = alloca i32, align 4
// CHECK:   [[_COMPOUNDLITERAL_I_I:%.*]] = alloca <16 x i32>, align 64
// CHECK:   [[__M_ADDR_I:%.*]] = alloca i16, align 2
// CHECK:   [[__V_ADDR_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__M_ADDR:%.*]] = alloca i16, align 2
// CHECK:   [[__W_ADDR:%.*]] = alloca <8 x i64>, align 64
// CHECK:   store i16 %__M, i16* [[__M_ADDR]], align 2
// CHECK:   store <8 x i64> %__W, <8 x i64>* [[__W_ADDR]], align 64
// CHECK:   [[TMP0:%.*]] = load i16, i16* [[__M_ADDR]], align 2
// CHECK:   [[TMP1:%.*]] = load <8 x i64>, <8 x i64>* [[__W_ADDR]], align 64
// CHECK:   store i16 [[TMP0]], i16* [[__M_ADDR_I]], align 2
// CHECK:   store <8 x i64> [[TMP1]], <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP2:%.*]] = load i16, i16* [[__M_ADDR_I]], align 2
// CHECK:   [[TMP3:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP4:%.*]] = bitcast <8 x i64> [[TMP3]] to <16 x i32>
// CHECK:   store i32 -2147483648, i32* [[__S_ADDR_I_I]], align 4
// CHECK:   [[TMP5:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK:   [[VECINIT_I_I:%.*]] = insertelement <16 x i32> undef, i32 [[TMP5]], i32 0
// CHECK:   [[TMP6:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK:   [[VECINIT1_I_I:%.*]] = insertelement <16 x i32> [[VECINIT_I_I]], i32 [[TMP6]], i32 1
// CHECK:   [[TMP7:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK:   [[VECINIT2_I_I:%.*]] = insertelement <16 x i32> [[VECINIT1_I_I]], i32 [[TMP7]], i32 2
// CHECK:   [[TMP8:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK:   [[VECINIT3_I_I:%.*]] = insertelement <16 x i32> [[VECINIT2_I_I]], i32 [[TMP8]], i32 3
// CHECK:   [[TMP9:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK:   [[VECINIT4_I_I:%.*]] = insertelement <16 x i32> [[VECINIT3_I_I]], i32 [[TMP9]], i32 4
// CHECK:   [[TMP10:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK:   [[VECINIT5_I_I:%.*]] = insertelement <16 x i32> [[VECINIT4_I_I]], i32 [[TMP10]], i32 5
// CHECK:   [[TMP11:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK:   [[VECINIT6_I_I:%.*]] = insertelement <16 x i32> [[VECINIT5_I_I]], i32 [[TMP11]], i32 6
// CHECK:   [[TMP12:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK:   [[VECINIT7_I_I:%.*]] = insertelement <16 x i32> [[VECINIT6_I_I]], i32 [[TMP12]], i32 7
// CHECK:   [[TMP13:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK:   [[VECINIT8_I_I:%.*]] = insertelement <16 x i32> [[VECINIT7_I_I]], i32 [[TMP13]], i32 8
// CHECK:   [[TMP14:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK:   [[VECINIT9_I_I:%.*]] = insertelement <16 x i32> [[VECINIT8_I_I]], i32 [[TMP14]], i32 9
// CHECK:   [[TMP15:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK:   [[VECINIT10_I_I:%.*]] = insertelement <16 x i32> [[VECINIT9_I_I]], i32 [[TMP15]], i32 10
// CHECK:   [[TMP16:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK:   [[VECINIT11_I_I:%.*]] = insertelement <16 x i32> [[VECINIT10_I_I]], i32 [[TMP16]], i32 11
// CHECK:   [[TMP17:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK:   [[VECINIT12_I_I:%.*]] = insertelement <16 x i32> [[VECINIT11_I_I]], i32 [[TMP17]], i32 12
// CHECK:   [[TMP18:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK:   [[VECINIT13_I_I:%.*]] = insertelement <16 x i32> [[VECINIT12_I_I]], i32 [[TMP18]], i32 13
// CHECK:   [[TMP19:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK:   [[VECINIT14_I_I:%.*]] = insertelement <16 x i32> [[VECINIT13_I_I]], i32 [[TMP19]], i32 14
// CHECK:   [[TMP20:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK:   [[VECINIT15_I_I:%.*]] = insertelement <16 x i32> [[VECINIT14_I_I]], i32 [[TMP20]], i32 15
// CHECK:   store <16 x i32> [[VECINIT15_I_I]], <16 x i32>* [[_COMPOUNDLITERAL_I_I]], align 64
// CHECK:   [[TMP21:%.*]] = load <16 x i32>, <16 x i32>* [[_COMPOUNDLITERAL_I_I]], align 64
// CHECK:   [[TMP22:%.*]] = bitcast <16 x i32> [[TMP21]] to <8 x i64>
// CHECK:   [[TMP23:%.*]] = bitcast i16 [[TMP2]] to <16 x i1>
// CHECK:   [[TMP24:%.*]] = select <16 x i1> [[TMP23]], <16 x i32> [[TMP4]], <16 x i32> [[TMP21]]
// CHECK:   [[TMP25:%.*]] = bitcast <16 x i32> [[TMP24]] to <8 x i64>
// CHECK:   store <8 x i64> [[TMP25]], <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP26:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP27:%.*]] = bitcast <8 x i64> [[TMP26]] to <16 x i32>
// CHECK:   [[TMP28:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP29:%.*]] = bitcast <8 x i64> [[TMP28]] to <16 x i32>
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <16 x i32> [[TMP27]], <16 x i32> [[TMP29]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP30:%.*]] = bitcast <16 x i32> [[SHUFFLE_I]] to <8 x i64>
// CHECK:   [[TMP31:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP32:%.*]] = bitcast <8 x i64> [[TMP31]] to <16 x i32>
// CHECK:   [[TMP33:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP34:%.*]] = bitcast <8 x i64> [[TMP33]] to <16 x i32>
// CHECK:   [[SHUFFLE1_I:%.*]] = shufflevector <16 x i32> [[TMP32]], <16 x i32> [[TMP34]], <16 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP35:%.*]] = bitcast <16 x i32> [[SHUFFLE1_I]] to <8 x i64>
// CHECK:   store <8 x i64> [[TMP30]], <8 x i64>* [[__A_ADDR_I19_I]], align 64
// CHECK:   store <8 x i64> [[TMP35]], <8 x i64>* [[__B_ADDR_I20_I]], align 64
// CHECK:   [[TMP36:%.*]] = load <8 x i64>, <8 x i64>* [[__A_ADDR_I19_I]], align 64
// CHECK:   [[TMP37:%.*]] = bitcast <8 x i64> [[TMP36]] to <16 x i32>
// CHECK:   [[TMP38:%.*]] = load <8 x i64>, <8 x i64>* [[__B_ADDR_I20_I]], align 64
// CHECK:   [[TMP39:%.*]] = bitcast <8 x i64> [[TMP38]] to <16 x i32>
// CHECK:   [[TMP42:%.*]] = icmp sgt <16 x i32> [[TMP37]], [[TMP39]]
// CHECK:   [[TMP43:%.*]] = select <16 x i1> [[TMP42]], <16 x i32> [[TMP37]], <16 x i32> [[TMP39]]
// CHECK:   [[TMP44:%.*]] = bitcast <16 x i32> [[TMP43]] to <8 x i64>
// CHECK:   store <8 x i64> [[TMP44]], <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP45:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP46:%.*]] = bitcast <8 x i64> [[TMP45]] to <16 x i32>
// CHECK:   [[TMP47:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP48:%.*]] = bitcast <8 x i64> [[TMP47]] to <16 x i32>
// CHECK:   [[SHUFFLE3_I:%.*]] = shufflevector <16 x i32> [[TMP46]], <16 x i32> [[TMP48]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP49:%.*]] = bitcast <16 x i32> [[SHUFFLE3_I]] to <8 x i64>
// CHECK:   [[TMP50:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP51:%.*]] = bitcast <8 x i64> [[TMP50]] to <16 x i32>
// CHECK:   [[TMP52:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP53:%.*]] = bitcast <8 x i64> [[TMP52]] to <16 x i32>
// CHECK:   [[SHUFFLE4_I:%.*]] = shufflevector <16 x i32> [[TMP51]], <16 x i32> [[TMP53]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP54:%.*]] = bitcast <16 x i32> [[SHUFFLE4_I]] to <8 x i64>
// CHECK:   store <8 x i64> [[TMP49]], <8 x i64>* [[__A_ADDR_I16_I]], align 64
// CHECK:   store <8 x i64> [[TMP54]], <8 x i64>* [[__B_ADDR_I17_I]], align 64
// CHECK:   [[TMP55:%.*]] = load <8 x i64>, <8 x i64>* [[__A_ADDR_I16_I]], align 64
// CHECK:   [[TMP56:%.*]] = bitcast <8 x i64> [[TMP55]] to <16 x i32>
// CHECK:   [[TMP57:%.*]] = load <8 x i64>, <8 x i64>* [[__B_ADDR_I17_I]], align 64
// CHECK:   [[TMP58:%.*]] = bitcast <8 x i64> [[TMP57]] to <16 x i32>
// CHECK:   [[TMP61:%.*]] = icmp sgt <16 x i32> [[TMP56]], [[TMP58]]
// CHECK:   [[TMP62:%.*]] = select <16 x i1> [[TMP61]], <16 x i32> [[TMP56]], <16 x i32> [[TMP58]]
// CHECK:   [[TMP63:%.*]] = bitcast <16 x i32> [[TMP62]] to <8 x i64>
// CHECK:   store <8 x i64> [[TMP63]], <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP64:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP65:%.*]] = bitcast <8 x i64> [[TMP64]] to <16 x i32>
// CHECK:   [[TMP66:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP67:%.*]] = bitcast <8 x i64> [[TMP66]] to <16 x i32>
// CHECK:   [[SHUFFLE6_I:%.*]] = shufflevector <16 x i32> [[TMP65]], <16 x i32> [[TMP67]], <16 x i32> <i32 0, i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP68:%.*]] = bitcast <16 x i32> [[SHUFFLE6_I]] to <8 x i64>
// CHECK:   [[TMP69:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP70:%.*]] = bitcast <8 x i64> [[TMP69]] to <16 x i32>
// CHECK:   [[TMP71:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP72:%.*]] = bitcast <8 x i64> [[TMP71]] to <16 x i32>
// CHECK:   [[SHUFFLE7_I:%.*]] = shufflevector <16 x i32> [[TMP70]], <16 x i32> [[TMP72]], <16 x i32> <i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP73:%.*]] = bitcast <16 x i32> [[SHUFFLE7_I]] to <8 x i64>
// CHECK:   store <8 x i64> [[TMP68]], <8 x i64>* [[__A_ADDR_I13_I]], align 64
// CHECK:   store <8 x i64> [[TMP73]], <8 x i64>* [[__B_ADDR_I14_I]], align 64
// CHECK:   [[TMP74:%.*]] = load <8 x i64>, <8 x i64>* [[__A_ADDR_I13_I]], align 64
// CHECK:   [[TMP75:%.*]] = bitcast <8 x i64> [[TMP74]] to <16 x i32>
// CHECK:   [[TMP76:%.*]] = load <8 x i64>, <8 x i64>* [[__B_ADDR_I14_I]], align 64
// CHECK:   [[TMP77:%.*]] = bitcast <8 x i64> [[TMP76]] to <16 x i32>
// CHECK:   [[TMP80:%.*]] = icmp sgt <16 x i32> [[TMP75]], [[TMP77]]
// CHECK:   [[TMP81:%.*]] = select <16 x i1> [[TMP80]], <16 x i32> [[TMP75]], <16 x i32> [[TMP77]]
// CHECK:   [[TMP82:%.*]] = bitcast <16 x i32> [[TMP81]] to <8 x i64>
// CHECK:   store <8 x i64> [[TMP82]], <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP83:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP84:%.*]] = bitcast <8 x i64> [[TMP83]] to <16 x i32>
// CHECK:   [[TMP85:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP86:%.*]] = bitcast <8 x i64> [[TMP85]] to <16 x i32>
// CHECK:   [[SHUFFLE9_I:%.*]] = shufflevector <16 x i32> [[TMP84]], <16 x i32> [[TMP86]], <16 x i32> <i32 0, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP87:%.*]] = bitcast <16 x i32> [[SHUFFLE9_I]] to <8 x i64>
// CHECK:   [[TMP88:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP89:%.*]] = bitcast <8 x i64> [[TMP88]] to <16 x i32>
// CHECK:   [[TMP90:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP91:%.*]] = bitcast <8 x i64> [[TMP90]] to <16 x i32>
// CHECK:   [[SHUFFLE10_I:%.*]] = shufflevector <16 x i32> [[TMP89]], <16 x i32> [[TMP91]], <16 x i32> <i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP92:%.*]] = bitcast <16 x i32> [[SHUFFLE10_I]] to <8 x i64>
// CHECK:   store <8 x i64> [[TMP87]], <8 x i64>* [[__A_ADDR_I_I]], align 64
// CHECK:   store <8 x i64> [[TMP92]], <8 x i64>* [[__B_ADDR_I_I]], align 64
// CHECK:   [[TMP93:%.*]] = load <8 x i64>, <8 x i64>* [[__A_ADDR_I_I]], align 64
// CHECK:   [[TMP94:%.*]] = bitcast <8 x i64> [[TMP93]] to <16 x i32>
// CHECK:   [[TMP95:%.*]] = load <8 x i64>, <8 x i64>* [[__B_ADDR_I_I]], align 64
// CHECK:   [[TMP96:%.*]] = bitcast <8 x i64> [[TMP95]] to <16 x i32>
// CHECK:   [[TMP99:%.*]] = icmp sgt <16 x i32> [[TMP94]], [[TMP96]]
// CHECK:   [[TMP100:%.*]] = select <16 x i1> [[TMP99]], <16 x i32> [[TMP94]], <16 x i32> [[TMP96]]
// CHECK:   [[TMP101:%.*]] = bitcast <16 x i32> [[TMP100]] to <8 x i64>
// CHECK:   store <8 x i64> [[TMP101]], <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP102:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[VECEXT_I:%.*]] = extractelement <8 x i64> [[TMP102]], i32 0
// CHECK:   [[CONV_I:%.*]] = trunc i64 [[VECEXT_I]] to i32
// CHECK:   ret i32 [[CONV_I]]
int test_mm512_mask_reduce_max_epi32(__mmask16 __M, __m512i __W){
  return _mm512_mask_reduce_max_epi32(__M, __W); 
}

// CHECK-LABEL: define i32 @test_mm512_mask_reduce_max_epu32(i16 zeroext %__M, <8 x i64> %__W) #0 {
// CHECK:   [[__A_ADDR_I19_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__B_ADDR_I20_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__A_ADDR_I16_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__B_ADDR_I17_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__A_ADDR_I13_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__B_ADDR_I14_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__A_ADDR_I_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__B_ADDR_I_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__S_ADDR_I_I:%.*]] = alloca i32, align 4
// CHECK:   [[_COMPOUNDLITERAL_I_I:%.*]] = alloca <16 x i32>, align 64
// CHECK:   [[__M_ADDR_I:%.*]] = alloca i16, align 2
// CHECK:   [[__V_ADDR_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__M_ADDR:%.*]] = alloca i16, align 2
// CHECK:   [[__W_ADDR:%.*]] = alloca <8 x i64>, align 64
// CHECK:   store i16 %__M, i16* [[__M_ADDR]], align 2
// CHECK:   store <8 x i64> %__W, <8 x i64>* [[__W_ADDR]], align 64
// CHECK:   [[TMP0:%.*]] = load i16, i16* [[__M_ADDR]], align 2
// CHECK:   [[TMP1:%.*]] = load <8 x i64>, <8 x i64>* [[__W_ADDR]], align 64
// CHECK:   store i16 [[TMP0]], i16* [[__M_ADDR_I]], align 2
// CHECK:   store <8 x i64> [[TMP1]], <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP2:%.*]] = load i16, i16* [[__M_ADDR_I]], align 2
// CHECK:   [[TMP3:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP4:%.*]] = bitcast <8 x i64> [[TMP3]] to <16 x i32>
// CHECK:   store i32 0, i32* [[__S_ADDR_I_I]], align 4
// CHECK:   [[TMP5:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK:   [[VECINIT_I_I:%.*]] = insertelement <16 x i32> undef, i32 [[TMP5]], i32 0
// CHECK:   [[TMP6:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK:   [[VECINIT1_I_I:%.*]] = insertelement <16 x i32> [[VECINIT_I_I]], i32 [[TMP6]], i32 1
// CHECK:   [[TMP7:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK:   [[VECINIT2_I_I:%.*]] = insertelement <16 x i32> [[VECINIT1_I_I]], i32 [[TMP7]], i32 2
// CHECK:   [[TMP8:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK:   [[VECINIT3_I_I:%.*]] = insertelement <16 x i32> [[VECINIT2_I_I]], i32 [[TMP8]], i32 3
// CHECK:   [[TMP9:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK:   [[VECINIT4_I_I:%.*]] = insertelement <16 x i32> [[VECINIT3_I_I]], i32 [[TMP9]], i32 4
// CHECK:   [[TMP10:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK:   [[VECINIT5_I_I:%.*]] = insertelement <16 x i32> [[VECINIT4_I_I]], i32 [[TMP10]], i32 5
// CHECK:   [[TMP11:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK:   [[VECINIT6_I_I:%.*]] = insertelement <16 x i32> [[VECINIT5_I_I]], i32 [[TMP11]], i32 6
// CHECK:   [[TMP12:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK:   [[VECINIT7_I_I:%.*]] = insertelement <16 x i32> [[VECINIT6_I_I]], i32 [[TMP12]], i32 7
// CHECK:   [[TMP13:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK:   [[VECINIT8_I_I:%.*]] = insertelement <16 x i32> [[VECINIT7_I_I]], i32 [[TMP13]], i32 8
// CHECK:   [[TMP14:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK:   [[VECINIT9_I_I:%.*]] = insertelement <16 x i32> [[VECINIT8_I_I]], i32 [[TMP14]], i32 9
// CHECK:   [[TMP15:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK:   [[VECINIT10_I_I:%.*]] = insertelement <16 x i32> [[VECINIT9_I_I]], i32 [[TMP15]], i32 10
// CHECK:   [[TMP16:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK:   [[VECINIT11_I_I:%.*]] = insertelement <16 x i32> [[VECINIT10_I_I]], i32 [[TMP16]], i32 11
// CHECK:   [[TMP17:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK:   [[VECINIT12_I_I:%.*]] = insertelement <16 x i32> [[VECINIT11_I_I]], i32 [[TMP17]], i32 12
// CHECK:   [[TMP18:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK:   [[VECINIT13_I_I:%.*]] = insertelement <16 x i32> [[VECINIT12_I_I]], i32 [[TMP18]], i32 13
// CHECK:   [[TMP19:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK:   [[VECINIT14_I_I:%.*]] = insertelement <16 x i32> [[VECINIT13_I_I]], i32 [[TMP19]], i32 14
// CHECK:   [[TMP20:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK:   [[VECINIT15_I_I:%.*]] = insertelement <16 x i32> [[VECINIT14_I_I]], i32 [[TMP20]], i32 15
// CHECK:   store <16 x i32> [[VECINIT15_I_I]], <16 x i32>* [[_COMPOUNDLITERAL_I_I]], align 64
// CHECK:   [[TMP21:%.*]] = load <16 x i32>, <16 x i32>* [[_COMPOUNDLITERAL_I_I]], align 64
// CHECK:   [[TMP22:%.*]] = bitcast <16 x i32> [[TMP21]] to <8 x i64>
// CHECK:   [[TMP23:%.*]] = bitcast i16 [[TMP2]] to <16 x i1>
// CHECK:   [[TMP24:%.*]] = select <16 x i1> [[TMP23]], <16 x i32> [[TMP4]], <16 x i32> [[TMP21]]
// CHECK:   [[TMP25:%.*]] = bitcast <16 x i32> [[TMP24]] to <8 x i64>
// CHECK:   store <8 x i64> [[TMP25]], <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP26:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP27:%.*]] = bitcast <8 x i64> [[TMP26]] to <16 x i32>
// CHECK:   [[TMP28:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP29:%.*]] = bitcast <8 x i64> [[TMP28]] to <16 x i32>
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <16 x i32> [[TMP27]], <16 x i32> [[TMP29]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP30:%.*]] = bitcast <16 x i32> [[SHUFFLE_I]] to <8 x i64>
// CHECK:   [[TMP31:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP32:%.*]] = bitcast <8 x i64> [[TMP31]] to <16 x i32>
// CHECK:   [[TMP33:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP34:%.*]] = bitcast <8 x i64> [[TMP33]] to <16 x i32>
// CHECK:   [[SHUFFLE1_I:%.*]] = shufflevector <16 x i32> [[TMP32]], <16 x i32> [[TMP34]], <16 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP35:%.*]] = bitcast <16 x i32> [[SHUFFLE1_I]] to <8 x i64>
// CHECK:   store <8 x i64> [[TMP30]], <8 x i64>* [[__A_ADDR_I19_I]], align 64
// CHECK:   store <8 x i64> [[TMP35]], <8 x i64>* [[__B_ADDR_I20_I]], align 64
// CHECK:   [[TMP36:%.*]] = load <8 x i64>, <8 x i64>* [[__A_ADDR_I19_I]], align 64
// CHECK:   [[TMP37:%.*]] = bitcast <8 x i64> [[TMP36]] to <16 x i32>
// CHECK:   [[TMP38:%.*]] = load <8 x i64>, <8 x i64>* [[__B_ADDR_I20_I]], align 64
// CHECK:   [[TMP39:%.*]] = bitcast <8 x i64> [[TMP38]] to <16 x i32>
// CHECK:   [[TMP42:%.*]] = icmp ugt <16 x i32> [[TMP37]], [[TMP39]]
// CHECK:   [[TMP43:%.*]] = select <16 x i1> [[TMP42]], <16 x i32> [[TMP37]], <16 x i32> [[TMP39]]
// CHECK:   [[TMP44:%.*]] = bitcast <16 x i32> [[TMP43]] to <8 x i64>
// CHECK:   store <8 x i64> [[TMP44]], <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP45:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP46:%.*]] = bitcast <8 x i64> [[TMP45]] to <16 x i32>
// CHECK:   [[TMP47:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP48:%.*]] = bitcast <8 x i64> [[TMP47]] to <16 x i32>
// CHECK:   [[SHUFFLE3_I:%.*]] = shufflevector <16 x i32> [[TMP46]], <16 x i32> [[TMP48]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP49:%.*]] = bitcast <16 x i32> [[SHUFFLE3_I]] to <8 x i64>
// CHECK:   [[TMP50:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP51:%.*]] = bitcast <8 x i64> [[TMP50]] to <16 x i32>
// CHECK:   [[TMP52:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP53:%.*]] = bitcast <8 x i64> [[TMP52]] to <16 x i32>
// CHECK:   [[SHUFFLE4_I:%.*]] = shufflevector <16 x i32> [[TMP51]], <16 x i32> [[TMP53]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP54:%.*]] = bitcast <16 x i32> [[SHUFFLE4_I]] to <8 x i64>
// CHECK:   store <8 x i64> [[TMP49]], <8 x i64>* [[__A_ADDR_I16_I]], align 64
// CHECK:   store <8 x i64> [[TMP54]], <8 x i64>* [[__B_ADDR_I17_I]], align 64
// CHECK:   [[TMP55:%.*]] = load <8 x i64>, <8 x i64>* [[__A_ADDR_I16_I]], align 64
// CHECK:   [[TMP56:%.*]] = bitcast <8 x i64> [[TMP55]] to <16 x i32>
// CHECK:   [[TMP57:%.*]] = load <8 x i64>, <8 x i64>* [[__B_ADDR_I17_I]], align 64
// CHECK:   [[TMP58:%.*]] = bitcast <8 x i64> [[TMP57]] to <16 x i32>
// CHECK:   [[TMP61:%.*]] = icmp ugt <16 x i32> [[TMP56]], [[TMP58]]
// CHECK:   [[TMP62:%.*]] = select <16 x i1> [[TMP61]], <16 x i32> [[TMP56]], <16 x i32> [[TMP58]]
// CHECK:   [[TMP63:%.*]] = bitcast <16 x i32> [[TMP62]] to <8 x i64>
// CHECK:   store <8 x i64> [[TMP63]], <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP64:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP65:%.*]] = bitcast <8 x i64> [[TMP64]] to <16 x i32>
// CHECK:   [[TMP66:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP67:%.*]] = bitcast <8 x i64> [[TMP66]] to <16 x i32>
// CHECK:   [[SHUFFLE6_I:%.*]] = shufflevector <16 x i32> [[TMP65]], <16 x i32> [[TMP67]], <16 x i32> <i32 0, i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP68:%.*]] = bitcast <16 x i32> [[SHUFFLE6_I]] to <8 x i64>
// CHECK:   [[TMP69:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP70:%.*]] = bitcast <8 x i64> [[TMP69]] to <16 x i32>
// CHECK:   [[TMP71:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP72:%.*]] = bitcast <8 x i64> [[TMP71]] to <16 x i32>
// CHECK:   [[SHUFFLE7_I:%.*]] = shufflevector <16 x i32> [[TMP70]], <16 x i32> [[TMP72]], <16 x i32> <i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP73:%.*]] = bitcast <16 x i32> [[SHUFFLE7_I]] to <8 x i64>
// CHECK:   store <8 x i64> [[TMP68]], <8 x i64>* [[__A_ADDR_I13_I]], align 64
// CHECK:   store <8 x i64> [[TMP73]], <8 x i64>* [[__B_ADDR_I14_I]], align 64
// CHECK:   [[TMP74:%.*]] = load <8 x i64>, <8 x i64>* [[__A_ADDR_I13_I]], align 64
// CHECK:   [[TMP75:%.*]] = bitcast <8 x i64> [[TMP74]] to <16 x i32>
// CHECK:   [[TMP76:%.*]] = load <8 x i64>, <8 x i64>* [[__B_ADDR_I14_I]], align 64
// CHECK:   [[TMP77:%.*]] = bitcast <8 x i64> [[TMP76]] to <16 x i32>
// CHECK:   [[TMP80:%.*]] = icmp ugt <16 x i32> [[TMP75]], [[TMP77]]
// CHECK:   [[TMP81:%.*]] = select <16 x i1> [[TMP80]], <16 x i32> [[TMP75]], <16 x i32> [[TMP77]]
// CHECK:   [[TMP82:%.*]] = bitcast <16 x i32> [[TMP81]] to <8 x i64>
// CHECK:   store <8 x i64> [[TMP82]], <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP83:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP84:%.*]] = bitcast <8 x i64> [[TMP83]] to <16 x i32>
// CHECK:   [[TMP85:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP86:%.*]] = bitcast <8 x i64> [[TMP85]] to <16 x i32>
// CHECK:   [[SHUFFLE9_I:%.*]] = shufflevector <16 x i32> [[TMP84]], <16 x i32> [[TMP86]], <16 x i32> <i32 0, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP87:%.*]] = bitcast <16 x i32> [[SHUFFLE9_I]] to <8 x i64>
// CHECK:   [[TMP88:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP89:%.*]] = bitcast <8 x i64> [[TMP88]] to <16 x i32>
// CHECK:   [[TMP90:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP91:%.*]] = bitcast <8 x i64> [[TMP90]] to <16 x i32>
// CHECK:   [[SHUFFLE10_I:%.*]] = shufflevector <16 x i32> [[TMP89]], <16 x i32> [[TMP91]], <16 x i32> <i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP92:%.*]] = bitcast <16 x i32> [[SHUFFLE10_I]] to <8 x i64>
// CHECK:   store <8 x i64> [[TMP87]], <8 x i64>* [[__A_ADDR_I_I]], align 64
// CHECK:   store <8 x i64> [[TMP92]], <8 x i64>* [[__B_ADDR_I_I]], align 64
// CHECK:   [[TMP93:%.*]] = load <8 x i64>, <8 x i64>* [[__A_ADDR_I_I]], align 64
// CHECK:   [[TMP94:%.*]] = bitcast <8 x i64> [[TMP93]] to <16 x i32>
// CHECK:   [[TMP95:%.*]] = load <8 x i64>, <8 x i64>* [[__B_ADDR_I_I]], align 64
// CHECK:   [[TMP96:%.*]] = bitcast <8 x i64> [[TMP95]] to <16 x i32>
// CHECK:   [[TMP99:%.*]] = icmp ugt <16 x i32> [[TMP94]], [[TMP96]]
// CHECK:   [[TMP100:%.*]] = select <16 x i1> [[TMP99]], <16 x i32> [[TMP94]], <16 x i32> [[TMP96]]
// CHECK:   [[TMP101:%.*]] = bitcast <16 x i32> [[TMP100]] to <8 x i64>
// CHECK:   store <8 x i64> [[TMP101]], <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP102:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[VECEXT_I:%.*]] = extractelement <8 x i64> [[TMP102]], i32 0
// CHECK:   [[CONV_I:%.*]] = trunc i64 [[VECEXT_I]] to i32
// CHECK:   ret i32 [[CONV_I]]
unsigned int test_mm512_mask_reduce_max_epu32(__mmask16 __M, __m512i __W){
  return _mm512_mask_reduce_max_epu32(__M, __W); 
}

// CHECK-LABEL: define float @test_mm512_mask_reduce_max_ps(i16 zeroext %__M, <16 x float> %__W) #0 {
// CHECK:   [[_COMPOUNDLITERAL_I_I18_I:%.*]] = alloca <16 x float>, align 64
// CHECK:   [[__A_ADDR_I19_I:%.*]] = alloca <16 x float>, align 64
// CHECK:   [[__B_ADDR_I20_I:%.*]] = alloca <16 x float>, align 64
// CHECK:   [[_COMPOUNDLITERAL_I_I15_I:%.*]] = alloca <16 x float>, align 64
// CHECK:   [[__A_ADDR_I16_I:%.*]] = alloca <16 x float>, align 64
// CHECK:   [[__B_ADDR_I17_I:%.*]] = alloca <16 x float>, align 64
// CHECK:   [[_COMPOUNDLITERAL_I_I12_I:%.*]] = alloca <16 x float>, align 64
// CHECK:   [[__A_ADDR_I13_I:%.*]] = alloca <16 x float>, align 64
// CHECK:   [[__B_ADDR_I14_I:%.*]] = alloca <16 x float>, align 64
// CHECK:   [[_COMPOUNDLITERAL_I_I_I:%.*]] = alloca <16 x float>, align 64
// CHECK:   [[__A_ADDR_I_I:%.*]] = alloca <16 x float>, align 64
// CHECK:   [[__B_ADDR_I_I:%.*]] = alloca <16 x float>, align 64
// CHECK:   [[__W_ADDR_I_I:%.*]] = alloca float, align 4
// CHECK:   [[_COMPOUNDLITERAL_I_I:%.*]] = alloca <16 x float>, align 64
// CHECK:   [[__M_ADDR_I:%.*]] = alloca i16, align 2
// CHECK:   [[__V_ADDR_I:%.*]] = alloca <16 x float>, align 64
// CHECK:   [[__M_ADDR:%.*]] = alloca i16, align 2
// CHECK:   [[__W_ADDR:%.*]] = alloca <16 x float>, align 64
// CHECK:   store i16 %__M, i16* [[__M_ADDR]], align 2
// CHECK:   store <16 x float> %__W, <16 x float>* [[__W_ADDR]], align 64
// CHECK:   [[TMP0:%.*]] = load i16, i16* [[__M_ADDR]], align 2
// CHECK:   [[TMP1:%.*]] = load <16 x float>, <16 x float>* [[__W_ADDR]], align 64
// CHECK:   store i16 [[TMP0]], i16* [[__M_ADDR_I]], align 2
// CHECK:   store <16 x float> [[TMP1]], <16 x float>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP2:%.*]] = load i16, i16* [[__M_ADDR_I]], align 2
// CHECK:   [[TMP3:%.*]] = load <16 x float>, <16 x float>* [[__V_ADDR_I]], align 64
// CHECK:   store float 0xFFF0000000000000, float* [[__W_ADDR_I_I]], align 4
// CHECK:   [[TMP4:%.*]] = load float, float* [[__W_ADDR_I_I]], align 4
// CHECK:   [[VECINIT_I_I:%.*]] = insertelement <16 x float> undef, float [[TMP4]], i32 0
// CHECK:   [[TMP5:%.*]] = load float, float* [[__W_ADDR_I_I]], align 4
// CHECK:   [[VECINIT1_I_I:%.*]] = insertelement <16 x float> [[VECINIT_I_I]], float [[TMP5]], i32 1
// CHECK:   [[TMP6:%.*]] = load float, float* [[__W_ADDR_I_I]], align 4
// CHECK:   [[VECINIT2_I_I:%.*]] = insertelement <16 x float> [[VECINIT1_I_I]], float [[TMP6]], i32 2
// CHECK:   [[TMP7:%.*]] = load float, float* [[__W_ADDR_I_I]], align 4
// CHECK:   [[VECINIT3_I_I:%.*]] = insertelement <16 x float> [[VECINIT2_I_I]], float [[TMP7]], i32 3
// CHECK:   [[TMP8:%.*]] = load float, float* [[__W_ADDR_I_I]], align 4
// CHECK:   [[VECINIT4_I_I:%.*]] = insertelement <16 x float> [[VECINIT3_I_I]], float [[TMP8]], i32 4
// CHECK:   [[TMP9:%.*]] = load float, float* [[__W_ADDR_I_I]], align 4
// CHECK:   [[VECINIT5_I_I:%.*]] = insertelement <16 x float> [[VECINIT4_I_I]], float [[TMP9]], i32 5
// CHECK:   [[TMP10:%.*]] = load float, float* [[__W_ADDR_I_I]], align 4
// CHECK:   [[VECINIT6_I_I:%.*]] = insertelement <16 x float> [[VECINIT5_I_I]], float [[TMP10]], i32 6
// CHECK:   [[TMP11:%.*]] = load float, float* [[__W_ADDR_I_I]], align 4
// CHECK:   [[VECINIT7_I_I:%.*]] = insertelement <16 x float> [[VECINIT6_I_I]], float [[TMP11]], i32 7
// CHECK:   [[TMP12:%.*]] = load float, float* [[__W_ADDR_I_I]], align 4
// CHECK:   [[VECINIT8_I_I:%.*]] = insertelement <16 x float> [[VECINIT7_I_I]], float [[TMP12]], i32 8
// CHECK:   [[TMP13:%.*]] = load float, float* [[__W_ADDR_I_I]], align 4
// CHECK:   [[VECINIT9_I_I:%.*]] = insertelement <16 x float> [[VECINIT8_I_I]], float [[TMP13]], i32 9
// CHECK:   [[TMP14:%.*]] = load float, float* [[__W_ADDR_I_I]], align 4
// CHECK:   [[VECINIT10_I_I:%.*]] = insertelement <16 x float> [[VECINIT9_I_I]], float [[TMP14]], i32 10
// CHECK:   [[TMP15:%.*]] = load float, float* [[__W_ADDR_I_I]], align 4
// CHECK:   [[VECINIT11_I_I:%.*]] = insertelement <16 x float> [[VECINIT10_I_I]], float [[TMP15]], i32 11
// CHECK:   [[TMP16:%.*]] = load float, float* [[__W_ADDR_I_I]], align 4
// CHECK:   [[VECINIT12_I_I:%.*]] = insertelement <16 x float> [[VECINIT11_I_I]], float [[TMP16]], i32 12
// CHECK:   [[TMP17:%.*]] = load float, float* [[__W_ADDR_I_I]], align 4
// CHECK:   [[VECINIT13_I_I:%.*]] = insertelement <16 x float> [[VECINIT12_I_I]], float [[TMP17]], i32 13
// CHECK:   [[TMP18:%.*]] = load float, float* [[__W_ADDR_I_I]], align 4
// CHECK:   [[VECINIT14_I_I:%.*]] = insertelement <16 x float> [[VECINIT13_I_I]], float [[TMP18]], i32 14
// CHECK:   [[TMP19:%.*]] = load float, float* [[__W_ADDR_I_I]], align 4
// CHECK:   [[VECINIT15_I_I:%.*]] = insertelement <16 x float> [[VECINIT14_I_I]], float [[TMP19]], i32 15
// CHECK:   store <16 x float> [[VECINIT15_I_I]], <16 x float>* [[_COMPOUNDLITERAL_I_I]], align 64
// CHECK:   [[TMP20:%.*]] = load <16 x float>, <16 x float>* [[_COMPOUNDLITERAL_I_I]], align 64
// CHECK:   [[TMP21:%.*]] = bitcast i16 [[TMP2]] to <16 x i1>
// CHECK:   [[TMP22:%.*]] = select <16 x i1> [[TMP21]], <16 x float> [[TMP3]], <16 x float> [[TMP20]]
// CHECK:   store <16 x float> [[TMP22]], <16 x float>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP23:%.*]] = load <16 x float>, <16 x float>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP24:%.*]] = load <16 x float>, <16 x float>* [[__V_ADDR_I]], align 64
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <16 x float> [[TMP23]], <16 x float> [[TMP24]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP25:%.*]] = load <16 x float>, <16 x float>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP26:%.*]] = load <16 x float>, <16 x float>* [[__V_ADDR_I]], align 64
// CHECK:   [[SHUFFLE1_I:%.*]] = shufflevector <16 x float> [[TMP25]], <16 x float> [[TMP26]], <16 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   store <16 x float> [[SHUFFLE_I]], <16 x float>* [[__A_ADDR_I19_I]], align 64
// CHECK:   store <16 x float> [[SHUFFLE1_I]], <16 x float>* [[__B_ADDR_I20_I]], align 64
// CHECK:   [[TMP27:%.*]] = load <16 x float>, <16 x float>* [[__A_ADDR_I19_I]], align 64
// CHECK:   [[TMP28:%.*]] = load <16 x float>, <16 x float>* [[__B_ADDR_I20_I]], align 64
// CHECK:   store <16 x float> zeroinitializer, <16 x float>* [[_COMPOUNDLITERAL_I_I18_I]], align 64
// CHECK:   [[TMP29:%.*]] = load <16 x float>, <16 x float>* [[_COMPOUNDLITERAL_I_I18_I]], align 64
// CHECK:   [[TMP30:%.*]] = call <16 x float> @llvm.x86.avx512.mask.max.ps.512(<16 x float> [[TMP27]], <16 x float> [[TMP28]], <16 x float> [[TMP29]], i16 -1, i32 4) #2
// CHECK:   store <16 x float> [[TMP30]], <16 x float>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP31:%.*]] = load <16 x float>, <16 x float>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP32:%.*]] = load <16 x float>, <16 x float>* [[__V_ADDR_I]], align 64
// CHECK:   [[SHUFFLE3_I:%.*]] = shufflevector <16 x float> [[TMP31]], <16 x float> [[TMP32]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP33:%.*]] = load <16 x float>, <16 x float>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP34:%.*]] = load <16 x float>, <16 x float>* [[__V_ADDR_I]], align 64
// CHECK:   [[SHUFFLE4_I:%.*]] = shufflevector <16 x float> [[TMP33]], <16 x float> [[TMP34]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   store <16 x float> [[SHUFFLE3_I]], <16 x float>* [[__A_ADDR_I16_I]], align 64
// CHECK:   store <16 x float> [[SHUFFLE4_I]], <16 x float>* [[__B_ADDR_I17_I]], align 64
// CHECK:   [[TMP35:%.*]] = load <16 x float>, <16 x float>* [[__A_ADDR_I16_I]], align 64
// CHECK:   [[TMP36:%.*]] = load <16 x float>, <16 x float>* [[__B_ADDR_I17_I]], align 64
// CHECK:   store <16 x float> zeroinitializer, <16 x float>* [[_COMPOUNDLITERAL_I_I15_I]], align 64
// CHECK:   [[TMP37:%.*]] = load <16 x float>, <16 x float>* [[_COMPOUNDLITERAL_I_I15_I]], align 64
// CHECK:   [[TMP38:%.*]] = call <16 x float> @llvm.x86.avx512.mask.max.ps.512(<16 x float> [[TMP35]], <16 x float> [[TMP36]], <16 x float> [[TMP37]], i16 -1, i32 4) #2
// CHECK:   store <16 x float> [[TMP38]], <16 x float>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP39:%.*]] = load <16 x float>, <16 x float>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP40:%.*]] = load <16 x float>, <16 x float>* [[__V_ADDR_I]], align 64
// CHECK:   [[SHUFFLE6_I:%.*]] = shufflevector <16 x float> [[TMP39]], <16 x float> [[TMP40]], <16 x i32> <i32 0, i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP41:%.*]] = load <16 x float>, <16 x float>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP42:%.*]] = load <16 x float>, <16 x float>* [[__V_ADDR_I]], align 64
// CHECK:   [[SHUFFLE7_I:%.*]] = shufflevector <16 x float> [[TMP41]], <16 x float> [[TMP42]], <16 x i32> <i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   store <16 x float> [[SHUFFLE6_I]], <16 x float>* [[__A_ADDR_I13_I]], align 64
// CHECK:   store <16 x float> [[SHUFFLE7_I]], <16 x float>* [[__B_ADDR_I14_I]], align 64
// CHECK:   [[TMP43:%.*]] = load <16 x float>, <16 x float>* [[__A_ADDR_I13_I]], align 64
// CHECK:   [[TMP44:%.*]] = load <16 x float>, <16 x float>* [[__B_ADDR_I14_I]], align 64
// CHECK:   store <16 x float> zeroinitializer, <16 x float>* [[_COMPOUNDLITERAL_I_I12_I]], align 64
// CHECK:   [[TMP45:%.*]] = load <16 x float>, <16 x float>* [[_COMPOUNDLITERAL_I_I12_I]], align 64
// CHECK:   [[TMP46:%.*]] = call <16 x float> @llvm.x86.avx512.mask.max.ps.512(<16 x float> [[TMP43]], <16 x float> [[TMP44]], <16 x float> [[TMP45]], i16 -1, i32 4) #2
// CHECK:   store <16 x float> [[TMP46]], <16 x float>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP47:%.*]] = load <16 x float>, <16 x float>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP48:%.*]] = load <16 x float>, <16 x float>* [[__V_ADDR_I]], align 64
// CHECK:   [[SHUFFLE9_I:%.*]] = shufflevector <16 x float> [[TMP47]], <16 x float> [[TMP48]], <16 x i32> <i32 0, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP49:%.*]] = load <16 x float>, <16 x float>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP50:%.*]] = load <16 x float>, <16 x float>* [[__V_ADDR_I]], align 64
// CHECK:   [[SHUFFLE10_I:%.*]] = shufflevector <16 x float> [[TMP49]], <16 x float> [[TMP50]], <16 x i32> <i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   store <16 x float> [[SHUFFLE9_I]], <16 x float>* [[__A_ADDR_I_I]], align 64
// CHECK:   store <16 x float> [[SHUFFLE10_I]], <16 x float>* [[__B_ADDR_I_I]], align 64
// CHECK:   [[TMP51:%.*]] = load <16 x float>, <16 x float>* [[__A_ADDR_I_I]], align 64
// CHECK:   [[TMP52:%.*]] = load <16 x float>, <16 x float>* [[__B_ADDR_I_I]], align 64
// CHECK:   store <16 x float> zeroinitializer, <16 x float>* [[_COMPOUNDLITERAL_I_I_I]], align 64
// CHECK:   [[TMP53:%.*]] = load <16 x float>, <16 x float>* [[_COMPOUNDLITERAL_I_I_I]], align 64
// CHECK:   [[TMP54:%.*]] = call <16 x float> @llvm.x86.avx512.mask.max.ps.512(<16 x float> [[TMP51]], <16 x float> [[TMP52]], <16 x float> [[TMP53]], i16 -1, i32 4) #2
// CHECK:   store <16 x float> [[TMP54]], <16 x float>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP55:%.*]] = load <16 x float>, <16 x float>* [[__V_ADDR_I]], align 64
// CHECK:   [[VECEXT_I:%.*]] = extractelement <16 x float> [[TMP55]], i32 0
// CHECK:   ret float [[VECEXT_I]]
float test_mm512_mask_reduce_max_ps(__mmask16 __M, __m512 __W){
  return _mm512_mask_reduce_max_ps(__M, __W); 
}

// CHECK-LABEL: define i32 @test_mm512_mask_reduce_min_epi32(i16 zeroext %__M, <8 x i64> %__W) #0 {
// CHECK:   [[__A_ADDR_I19_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__B_ADDR_I20_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__A_ADDR_I16_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__B_ADDR_I17_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__A_ADDR_I13_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__B_ADDR_I14_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__A_ADDR_I_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__B_ADDR_I_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__S_ADDR_I_I:%.*]] = alloca i32, align 4
// CHECK:   [[_COMPOUNDLITERAL_I_I:%.*]] = alloca <16 x i32>, align 64
// CHECK:   [[__M_ADDR_I:%.*]] = alloca i16, align 2
// CHECK:   [[__V_ADDR_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__M_ADDR:%.*]] = alloca i16, align 2
// CHECK:   [[__W_ADDR:%.*]] = alloca <8 x i64>, align 64
// CHECK:   store i16 %__M, i16* [[__M_ADDR]], align 2
// CHECK:   store <8 x i64> %__W, <8 x i64>* [[__W_ADDR]], align 64
// CHECK:   [[TMP0:%.*]] = load i16, i16* [[__M_ADDR]], align 2
// CHECK:   [[TMP1:%.*]] = load <8 x i64>, <8 x i64>* [[__W_ADDR]], align 64
// CHECK:   store i16 [[TMP0]], i16* [[__M_ADDR_I]], align 2
// CHECK:   store <8 x i64> [[TMP1]], <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP2:%.*]] = load i16, i16* [[__M_ADDR_I]], align 2
// CHECK:   [[TMP3:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP4:%.*]] = bitcast <8 x i64> [[TMP3]] to <16 x i32>
// CHECK:   store i32 2147483647, i32* [[__S_ADDR_I_I]], align 4
// CHECK:   [[TMP5:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK:   [[VECINIT_I_I:%.*]] = insertelement <16 x i32> undef, i32 [[TMP5]], i32 0
// CHECK:   [[TMP6:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK:   [[VECINIT1_I_I:%.*]] = insertelement <16 x i32> [[VECINIT_I_I]], i32 [[TMP6]], i32 1
// CHECK:   [[TMP7:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK:   [[VECINIT2_I_I:%.*]] = insertelement <16 x i32> [[VECINIT1_I_I]], i32 [[TMP7]], i32 2
// CHECK:   [[TMP8:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK:   [[VECINIT3_I_I:%.*]] = insertelement <16 x i32> [[VECINIT2_I_I]], i32 [[TMP8]], i32 3
// CHECK:   [[TMP9:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK:   [[VECINIT4_I_I:%.*]] = insertelement <16 x i32> [[VECINIT3_I_I]], i32 [[TMP9]], i32 4
// CHECK:   [[TMP10:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK:   [[VECINIT5_I_I:%.*]] = insertelement <16 x i32> [[VECINIT4_I_I]], i32 [[TMP10]], i32 5
// CHECK:   [[TMP11:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK:   [[VECINIT6_I_I:%.*]] = insertelement <16 x i32> [[VECINIT5_I_I]], i32 [[TMP11]], i32 6
// CHECK:   [[TMP12:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK:   [[VECINIT7_I_I:%.*]] = insertelement <16 x i32> [[VECINIT6_I_I]], i32 [[TMP12]], i32 7
// CHECK:   [[TMP13:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK:   [[VECINIT8_I_I:%.*]] = insertelement <16 x i32> [[VECINIT7_I_I]], i32 [[TMP13]], i32 8
// CHECK:   [[TMP14:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK:   [[VECINIT9_I_I:%.*]] = insertelement <16 x i32> [[VECINIT8_I_I]], i32 [[TMP14]], i32 9
// CHECK:   [[TMP15:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK:   [[VECINIT10_I_I:%.*]] = insertelement <16 x i32> [[VECINIT9_I_I]], i32 [[TMP15]], i32 10
// CHECK:   [[TMP16:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK:   [[VECINIT11_I_I:%.*]] = insertelement <16 x i32> [[VECINIT10_I_I]], i32 [[TMP16]], i32 11
// CHECK:   [[TMP17:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK:   [[VECINIT12_I_I:%.*]] = insertelement <16 x i32> [[VECINIT11_I_I]], i32 [[TMP17]], i32 12
// CHECK:   [[TMP18:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK:   [[VECINIT13_I_I:%.*]] = insertelement <16 x i32> [[VECINIT12_I_I]], i32 [[TMP18]], i32 13
// CHECK:   [[TMP19:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK:   [[VECINIT14_I_I:%.*]] = insertelement <16 x i32> [[VECINIT13_I_I]], i32 [[TMP19]], i32 14
// CHECK:   [[TMP20:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK:   [[VECINIT15_I_I:%.*]] = insertelement <16 x i32> [[VECINIT14_I_I]], i32 [[TMP20]], i32 15
// CHECK:   store <16 x i32> [[VECINIT15_I_I]], <16 x i32>* [[_COMPOUNDLITERAL_I_I]], align 64
// CHECK:   [[TMP21:%.*]] = load <16 x i32>, <16 x i32>* [[_COMPOUNDLITERAL_I_I]], align 64
// CHECK:   [[TMP22:%.*]] = bitcast <16 x i32> [[TMP21]] to <8 x i64>
// CHECK:   [[TMP23:%.*]] = bitcast i16 [[TMP2]] to <16 x i1>
// CHECK:   [[TMP24:%.*]] = select <16 x i1> [[TMP23]], <16 x i32> [[TMP4]], <16 x i32> [[TMP21]]
// CHECK:   [[TMP25:%.*]] = bitcast <16 x i32> [[TMP24]] to <8 x i64>
// CHECK:   store <8 x i64> [[TMP25]], <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP26:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP27:%.*]] = bitcast <8 x i64> [[TMP26]] to <16 x i32>
// CHECK:   [[TMP28:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP29:%.*]] = bitcast <8 x i64> [[TMP28]] to <16 x i32>
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <16 x i32> [[TMP27]], <16 x i32> [[TMP29]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP30:%.*]] = bitcast <16 x i32> [[SHUFFLE_I]] to <8 x i64>
// CHECK:   [[TMP31:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP32:%.*]] = bitcast <8 x i64> [[TMP31]] to <16 x i32>
// CHECK:   [[TMP33:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP34:%.*]] = bitcast <8 x i64> [[TMP33]] to <16 x i32>
// CHECK:   [[SHUFFLE1_I:%.*]] = shufflevector <16 x i32> [[TMP32]], <16 x i32> [[TMP34]], <16 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP35:%.*]] = bitcast <16 x i32> [[SHUFFLE1_I]] to <8 x i64>
// CHECK:   store <8 x i64> [[TMP30]], <8 x i64>* [[__A_ADDR_I19_I]], align 64
// CHECK:   store <8 x i64> [[TMP35]], <8 x i64>* [[__B_ADDR_I20_I]], align 64
// CHECK:   [[TMP36:%.*]] = load <8 x i64>, <8 x i64>* [[__A_ADDR_I19_I]], align 64
// CHECK:   [[TMP37:%.*]] = bitcast <8 x i64> [[TMP36]] to <16 x i32>
// CHECK:   [[TMP38:%.*]] = load <8 x i64>, <8 x i64>* [[__B_ADDR_I20_I]], align 64
// CHECK:   [[TMP39:%.*]] = bitcast <8 x i64> [[TMP38]] to <16 x i32>
// CHECK:   [[TMP42:%.*]] = icmp slt <16 x i32> [[TMP37]], [[TMP39]]
// CHECK:   [[TMP43:%.*]] = select <16 x i1> [[TMP42]], <16 x i32> [[TMP37]], <16 x i32> [[TMP39]]
// CHECK:   [[TMP44:%.*]] = bitcast <16 x i32> [[TMP43]] to <8 x i64>
// CHECK:   store <8 x i64> [[TMP44]], <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP45:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP46:%.*]] = bitcast <8 x i64> [[TMP45]] to <16 x i32>
// CHECK:   [[TMP47:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP48:%.*]] = bitcast <8 x i64> [[TMP47]] to <16 x i32>
// CHECK:   [[SHUFFLE3_I:%.*]] = shufflevector <16 x i32> [[TMP46]], <16 x i32> [[TMP48]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP49:%.*]] = bitcast <16 x i32> [[SHUFFLE3_I]] to <8 x i64>
// CHECK:   [[TMP50:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP51:%.*]] = bitcast <8 x i64> [[TMP50]] to <16 x i32>
// CHECK:   [[TMP52:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP53:%.*]] = bitcast <8 x i64> [[TMP52]] to <16 x i32>
// CHECK:   [[SHUFFLE4_I:%.*]] = shufflevector <16 x i32> [[TMP51]], <16 x i32> [[TMP53]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP54:%.*]] = bitcast <16 x i32> [[SHUFFLE4_I]] to <8 x i64>
// CHECK:   store <8 x i64> [[TMP49]], <8 x i64>* [[__A_ADDR_I16_I]], align 64
// CHECK:   store <8 x i64> [[TMP54]], <8 x i64>* [[__B_ADDR_I17_I]], align 64
// CHECK:   [[TMP55:%.*]] = load <8 x i64>, <8 x i64>* [[__A_ADDR_I16_I]], align 64
// CHECK:   [[TMP56:%.*]] = bitcast <8 x i64> [[TMP55]] to <16 x i32>
// CHECK:   [[TMP57:%.*]] = load <8 x i64>, <8 x i64>* [[__B_ADDR_I17_I]], align 64
// CHECK:   [[TMP58:%.*]] = bitcast <8 x i64> [[TMP57]] to <16 x i32>
// CHECK:   [[TMP61:%.*]] = icmp slt <16 x i32> [[TMP56]], [[TMP58]]
// CHECK:   [[TMP62:%.*]] = select <16 x i1> [[TMP61]], <16 x i32> [[TMP56]], <16 x i32> [[TMP58]]
// CHECK:   [[TMP63:%.*]] = bitcast <16 x i32> [[TMP62]] to <8 x i64>
// CHECK:   store <8 x i64> [[TMP63]], <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP64:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP65:%.*]] = bitcast <8 x i64> [[TMP64]] to <16 x i32>
// CHECK:   [[TMP66:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP67:%.*]] = bitcast <8 x i64> [[TMP66]] to <16 x i32>
// CHECK:   [[SHUFFLE6_I:%.*]] = shufflevector <16 x i32> [[TMP65]], <16 x i32> [[TMP67]], <16 x i32> <i32 0, i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP68:%.*]] = bitcast <16 x i32> [[SHUFFLE6_I]] to <8 x i64>
// CHECK:   [[TMP69:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP70:%.*]] = bitcast <8 x i64> [[TMP69]] to <16 x i32>
// CHECK:   [[TMP71:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP72:%.*]] = bitcast <8 x i64> [[TMP71]] to <16 x i32>
// CHECK:   [[SHUFFLE7_I:%.*]] = shufflevector <16 x i32> [[TMP70]], <16 x i32> [[TMP72]], <16 x i32> <i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP73:%.*]] = bitcast <16 x i32> [[SHUFFLE7_I]] to <8 x i64>
// CHECK:   store <8 x i64> [[TMP68]], <8 x i64>* [[__A_ADDR_I13_I]], align 64
// CHECK:   store <8 x i64> [[TMP73]], <8 x i64>* [[__B_ADDR_I14_I]], align 64
// CHECK:   [[TMP74:%.*]] = load <8 x i64>, <8 x i64>* [[__A_ADDR_I13_I]], align 64
// CHECK:   [[TMP75:%.*]] = bitcast <8 x i64> [[TMP74]] to <16 x i32>
// CHECK:   [[TMP76:%.*]] = load <8 x i64>, <8 x i64>* [[__B_ADDR_I14_I]], align 64
// CHECK:   [[TMP77:%.*]] = bitcast <8 x i64> [[TMP76]] to <16 x i32>
// CHECK:   [[TMP80:%.*]] = icmp slt <16 x i32> [[TMP75]], [[TMP77]]
// CHECK:   [[TMP81:%.*]] = select <16 x i1> [[TMP80]], <16 x i32> [[TMP75]], <16 x i32> [[TMP77]]
// CHECK:   [[TMP82:%.*]] = bitcast <16 x i32> [[TMP81]] to <8 x i64>
// CHECK:   store <8 x i64> [[TMP82]], <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP83:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP84:%.*]] = bitcast <8 x i64> [[TMP83]] to <16 x i32>
// CHECK:   [[TMP85:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP86:%.*]] = bitcast <8 x i64> [[TMP85]] to <16 x i32>
// CHECK:   [[SHUFFLE9_I:%.*]] = shufflevector <16 x i32> [[TMP84]], <16 x i32> [[TMP86]], <16 x i32> <i32 0, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP87:%.*]] = bitcast <16 x i32> [[SHUFFLE9_I]] to <8 x i64>
// CHECK:   [[TMP88:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP89:%.*]] = bitcast <8 x i64> [[TMP88]] to <16 x i32>
// CHECK:   [[TMP90:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP91:%.*]] = bitcast <8 x i64> [[TMP90]] to <16 x i32>
// CHECK:   [[SHUFFLE10_I:%.*]] = shufflevector <16 x i32> [[TMP89]], <16 x i32> [[TMP91]], <16 x i32> <i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP92:%.*]] = bitcast <16 x i32> [[SHUFFLE10_I]] to <8 x i64>
// CHECK:   store <8 x i64> [[TMP87]], <8 x i64>* [[__A_ADDR_I_I]], align 64
// CHECK:   store <8 x i64> [[TMP92]], <8 x i64>* [[__B_ADDR_I_I]], align 64
// CHECK:   [[TMP93:%.*]] = load <8 x i64>, <8 x i64>* [[__A_ADDR_I_I]], align 64
// CHECK:   [[TMP94:%.*]] = bitcast <8 x i64> [[TMP93]] to <16 x i32>
// CHECK:   [[TMP95:%.*]] = load <8 x i64>, <8 x i64>* [[__B_ADDR_I_I]], align 64
// CHECK:   [[TMP96:%.*]] = bitcast <8 x i64> [[TMP95]] to <16 x i32>
// CHECK:   [[TMP99:%.*]] = icmp slt <16 x i32> [[TMP94]], [[TMP96]]
// CHECK:   [[TMP100:%.*]] = select <16 x i1> [[TMP99]], <16 x i32> [[TMP94]], <16 x i32> [[TMP96]]
// CHECK:   [[TMP101:%.*]] = bitcast <16 x i32> [[TMP100]] to <8 x i64>
// CHECK:   store <8 x i64> [[TMP101]], <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP102:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[VECEXT_I:%.*]] = extractelement <8 x i64> [[TMP102]], i32 0
// CHECK:   [[CONV_I:%.*]] = trunc i64 [[VECEXT_I]] to i32
// CHECK:   ret i32 [[CONV_I]]
int test_mm512_mask_reduce_min_epi32(__mmask16 __M, __m512i __W){
  return _mm512_mask_reduce_min_epi32(__M, __W); 
}

// CHECK-LABEL: define i32 @test_mm512_mask_reduce_min_epu32(i16 zeroext %__M, <8 x i64> %__W) #0 {
// CHECK:   [[__A_ADDR_I19_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__B_ADDR_I20_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__A_ADDR_I16_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__B_ADDR_I17_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__A_ADDR_I13_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__B_ADDR_I14_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__A_ADDR_I_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__B_ADDR_I_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__S_ADDR_I_I:%.*]] = alloca i32, align 4
// CHECK:   [[_COMPOUNDLITERAL_I_I:%.*]] = alloca <16 x i32>, align 64
// CHECK:   [[__M_ADDR_I:%.*]] = alloca i16, align 2
// CHECK:   [[__V_ADDR_I:%.*]] = alloca <8 x i64>, align 64
// CHECK:   [[__M_ADDR:%.*]] = alloca i16, align 2
// CHECK:   [[__W_ADDR:%.*]] = alloca <8 x i64>, align 64
// CHECK:   store i16 %__M, i16* [[__M_ADDR]], align 2
// CHECK:   store <8 x i64> %__W, <8 x i64>* [[__W_ADDR]], align 64
// CHECK:   [[TMP0:%.*]] = load i16, i16* [[__M_ADDR]], align 2
// CHECK:   [[TMP1:%.*]] = load <8 x i64>, <8 x i64>* [[__W_ADDR]], align 64
// CHECK:   store i16 [[TMP0]], i16* [[__M_ADDR_I]], align 2
// CHECK:   store <8 x i64> [[TMP1]], <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP2:%.*]] = load i16, i16* [[__M_ADDR_I]], align 2
// CHECK:   [[TMP3:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP4:%.*]] = bitcast <8 x i64> [[TMP3]] to <16 x i32>
// CHECK:   store i32 -1, i32* [[__S_ADDR_I_I]], align 4
// CHECK:   [[TMP5:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK:   [[VECINIT_I_I:%.*]] = insertelement <16 x i32> undef, i32 [[TMP5]], i32 0
// CHECK:   [[TMP6:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK:   [[VECINIT1_I_I:%.*]] = insertelement <16 x i32> [[VECINIT_I_I]], i32 [[TMP6]], i32 1
// CHECK:   [[TMP7:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK:   [[VECINIT2_I_I:%.*]] = insertelement <16 x i32> [[VECINIT1_I_I]], i32 [[TMP7]], i32 2
// CHECK:   [[TMP8:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK:   [[VECINIT3_I_I:%.*]] = insertelement <16 x i32> [[VECINIT2_I_I]], i32 [[TMP8]], i32 3
// CHECK:   [[TMP9:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK:   [[VECINIT4_I_I:%.*]] = insertelement <16 x i32> [[VECINIT3_I_I]], i32 [[TMP9]], i32 4
// CHECK:   [[TMP10:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK:   [[VECINIT5_I_I:%.*]] = insertelement <16 x i32> [[VECINIT4_I_I]], i32 [[TMP10]], i32 5
// CHECK:   [[TMP11:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK:   [[VECINIT6_I_I:%.*]] = insertelement <16 x i32> [[VECINIT5_I_I]], i32 [[TMP11]], i32 6
// CHECK:   [[TMP12:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK:   [[VECINIT7_I_I:%.*]] = insertelement <16 x i32> [[VECINIT6_I_I]], i32 [[TMP12]], i32 7
// CHECK:   [[TMP13:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK:   [[VECINIT8_I_I:%.*]] = insertelement <16 x i32> [[VECINIT7_I_I]], i32 [[TMP13]], i32 8
// CHECK:   [[TMP14:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK:   [[VECINIT9_I_I:%.*]] = insertelement <16 x i32> [[VECINIT8_I_I]], i32 [[TMP14]], i32 9
// CHECK:   [[TMP15:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK:   [[VECINIT10_I_I:%.*]] = insertelement <16 x i32> [[VECINIT9_I_I]], i32 [[TMP15]], i32 10
// CHECK:   [[TMP16:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK:   [[VECINIT11_I_I:%.*]] = insertelement <16 x i32> [[VECINIT10_I_I]], i32 [[TMP16]], i32 11
// CHECK:   [[TMP17:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK:   [[VECINIT12_I_I:%.*]] = insertelement <16 x i32> [[VECINIT11_I_I]], i32 [[TMP17]], i32 12
// CHECK:   [[TMP18:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK:   [[VECINIT13_I_I:%.*]] = insertelement <16 x i32> [[VECINIT12_I_I]], i32 [[TMP18]], i32 13
// CHECK:   [[TMP19:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK:   [[VECINIT14_I_I:%.*]] = insertelement <16 x i32> [[VECINIT13_I_I]], i32 [[TMP19]], i32 14
// CHECK:   [[TMP20:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK:   [[VECINIT15_I_I:%.*]] = insertelement <16 x i32> [[VECINIT14_I_I]], i32 [[TMP20]], i32 15
// CHECK:   store <16 x i32> [[VECINIT15_I_I]], <16 x i32>* [[_COMPOUNDLITERAL_I_I]], align 64
// CHECK:   [[TMP21:%.*]] = load <16 x i32>, <16 x i32>* [[_COMPOUNDLITERAL_I_I]], align 64
// CHECK:   [[TMP22:%.*]] = bitcast <16 x i32> [[TMP21]] to <8 x i64>
// CHECK:   [[TMP23:%.*]] = bitcast i16 [[TMP2]] to <16 x i1>
// CHECK:   [[TMP24:%.*]] = select <16 x i1> [[TMP23]], <16 x i32> [[TMP4]], <16 x i32> [[TMP21]]
// CHECK:   [[TMP25:%.*]] = bitcast <16 x i32> [[TMP24]] to <8 x i64>
// CHECK:   store <8 x i64> [[TMP25]], <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP26:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP27:%.*]] = bitcast <8 x i64> [[TMP26]] to <16 x i32>
// CHECK:   [[TMP28:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP29:%.*]] = bitcast <8 x i64> [[TMP28]] to <16 x i32>
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <16 x i32> [[TMP27]], <16 x i32> [[TMP29]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP30:%.*]] = bitcast <16 x i32> [[SHUFFLE_I]] to <8 x i64>
// CHECK:   [[TMP31:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP32:%.*]] = bitcast <8 x i64> [[TMP31]] to <16 x i32>
// CHECK:   [[TMP33:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP34:%.*]] = bitcast <8 x i64> [[TMP33]] to <16 x i32>
// CHECK:   [[SHUFFLE1_I:%.*]] = shufflevector <16 x i32> [[TMP32]], <16 x i32> [[TMP34]], <16 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP35:%.*]] = bitcast <16 x i32> [[SHUFFLE1_I]] to <8 x i64>
// CHECK:   store <8 x i64> [[TMP30]], <8 x i64>* [[__A_ADDR_I19_I]], align 64
// CHECK:   store <8 x i64> [[TMP35]], <8 x i64>* [[__B_ADDR_I20_I]], align 64
// CHECK:   [[TMP36:%.*]] = load <8 x i64>, <8 x i64>* [[__A_ADDR_I19_I]], align 64
// CHECK:   [[TMP37:%.*]] = bitcast <8 x i64> [[TMP36]] to <16 x i32>
// CHECK:   [[TMP38:%.*]] = load <8 x i64>, <8 x i64>* [[__B_ADDR_I20_I]], align 64
// CHECK:   [[TMP39:%.*]] = bitcast <8 x i64> [[TMP38]] to <16 x i32>
// CHECK:   [[TMP42:%.*]] = icmp ult <16 x i32> [[TMP37]], [[TMP39]]
// CHECK:   [[TMP43:%.*]] = select <16 x i1> [[TMP42]], <16 x i32> [[TMP37]], <16 x i32> [[TMP39]]
// CHECK:   [[TMP44:%.*]] = bitcast <16 x i32> [[TMP43]] to <8 x i64>
// CHECK:   store <8 x i64> [[TMP44]], <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP45:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP46:%.*]] = bitcast <8 x i64> [[TMP45]] to <16 x i32>
// CHECK:   [[TMP47:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP48:%.*]] = bitcast <8 x i64> [[TMP47]] to <16 x i32>
// CHECK:   [[SHUFFLE3_I:%.*]] = shufflevector <16 x i32> [[TMP46]], <16 x i32> [[TMP48]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP49:%.*]] = bitcast <16 x i32> [[SHUFFLE3_I]] to <8 x i64>
// CHECK:   [[TMP50:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP51:%.*]] = bitcast <8 x i64> [[TMP50]] to <16 x i32>
// CHECK:   [[TMP52:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP53:%.*]] = bitcast <8 x i64> [[TMP52]] to <16 x i32>
// CHECK:   [[SHUFFLE4_I:%.*]] = shufflevector <16 x i32> [[TMP51]], <16 x i32> [[TMP53]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP54:%.*]] = bitcast <16 x i32> [[SHUFFLE4_I]] to <8 x i64>
// CHECK:   store <8 x i64> [[TMP49]], <8 x i64>* [[__A_ADDR_I16_I]], align 64
// CHECK:   store <8 x i64> [[TMP54]], <8 x i64>* [[__B_ADDR_I17_I]], align 64
// CHECK:   [[TMP55:%.*]] = load <8 x i64>, <8 x i64>* [[__A_ADDR_I16_I]], align 64
// CHECK:   [[TMP56:%.*]] = bitcast <8 x i64> [[TMP55]] to <16 x i32>
// CHECK:   [[TMP57:%.*]] = load <8 x i64>, <8 x i64>* [[__B_ADDR_I17_I]], align 64
// CHECK:   [[TMP58:%.*]] = bitcast <8 x i64> [[TMP57]] to <16 x i32>
// CHECK:   [[TMP61:%.*]] = icmp ult <16 x i32> [[TMP56]], [[TMP58]]
// CHECK:   [[TMP62:%.*]] = select <16 x i1> [[TMP61]], <16 x i32> [[TMP56]], <16 x i32> [[TMP58]]
// CHECK:   [[TMP63:%.*]] = bitcast <16 x i32> [[TMP62]] to <8 x i64>
// CHECK:   store <8 x i64> [[TMP63]], <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP64:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP65:%.*]] = bitcast <8 x i64> [[TMP64]] to <16 x i32>
// CHECK:   [[TMP66:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP67:%.*]] = bitcast <8 x i64> [[TMP66]] to <16 x i32>
// CHECK:   [[SHUFFLE6_I:%.*]] = shufflevector <16 x i32> [[TMP65]], <16 x i32> [[TMP67]], <16 x i32> <i32 0, i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP68:%.*]] = bitcast <16 x i32> [[SHUFFLE6_I]] to <8 x i64>
// CHECK:   [[TMP69:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP70:%.*]] = bitcast <8 x i64> [[TMP69]] to <16 x i32>
// CHECK:   [[TMP71:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP72:%.*]] = bitcast <8 x i64> [[TMP71]] to <16 x i32>
// CHECK:   [[SHUFFLE7_I:%.*]] = shufflevector <16 x i32> [[TMP70]], <16 x i32> [[TMP72]], <16 x i32> <i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP73:%.*]] = bitcast <16 x i32> [[SHUFFLE7_I]] to <8 x i64>
// CHECK:   store <8 x i64> [[TMP68]], <8 x i64>* [[__A_ADDR_I13_I]], align 64
// CHECK:   store <8 x i64> [[TMP73]], <8 x i64>* [[__B_ADDR_I14_I]], align 64
// CHECK:   [[TMP74:%.*]] = load <8 x i64>, <8 x i64>* [[__A_ADDR_I13_I]], align 64
// CHECK:   [[TMP75:%.*]] = bitcast <8 x i64> [[TMP74]] to <16 x i32>
// CHECK:   [[TMP76:%.*]] = load <8 x i64>, <8 x i64>* [[__B_ADDR_I14_I]], align 64
// CHECK:   [[TMP77:%.*]] = bitcast <8 x i64> [[TMP76]] to <16 x i32>
// CHECK:   [[TMP80:%.*]] = icmp ult <16 x i32> [[TMP75]], [[TMP77]]
// CHECK:   [[TMP81:%.*]] = select <16 x i1> [[TMP80]], <16 x i32> [[TMP75]], <16 x i32> [[TMP77]]
// CHECK:   [[TMP82:%.*]] = bitcast <16 x i32> [[TMP81]] to <8 x i64>
// CHECK:   store <8 x i64> [[TMP82]], <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP83:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP84:%.*]] = bitcast <8 x i64> [[TMP83]] to <16 x i32>
// CHECK:   [[TMP85:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP86:%.*]] = bitcast <8 x i64> [[TMP85]] to <16 x i32>
// CHECK:   [[SHUFFLE9_I:%.*]] = shufflevector <16 x i32> [[TMP84]], <16 x i32> [[TMP86]], <16 x i32> <i32 0, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP87:%.*]] = bitcast <16 x i32> [[SHUFFLE9_I]] to <8 x i64>
// CHECK:   [[TMP88:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP89:%.*]] = bitcast <8 x i64> [[TMP88]] to <16 x i32>
// CHECK:   [[TMP90:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP91:%.*]] = bitcast <8 x i64> [[TMP90]] to <16 x i32>
// CHECK:   [[SHUFFLE10_I:%.*]] = shufflevector <16 x i32> [[TMP89]], <16 x i32> [[TMP91]], <16 x i32> <i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP92:%.*]] = bitcast <16 x i32> [[SHUFFLE10_I]] to <8 x i64>
// CHECK:   store <8 x i64> [[TMP87]], <8 x i64>* [[__A_ADDR_I_I]], align 64
// CHECK:   store <8 x i64> [[TMP92]], <8 x i64>* [[__B_ADDR_I_I]], align 64
// CHECK:   [[TMP93:%.*]] = load <8 x i64>, <8 x i64>* [[__A_ADDR_I_I]], align 64
// CHECK:   [[TMP94:%.*]] = bitcast <8 x i64> [[TMP93]] to <16 x i32>
// CHECK:   [[TMP95:%.*]] = load <8 x i64>, <8 x i64>* [[__B_ADDR_I_I]], align 64
// CHECK:   [[TMP96:%.*]] = bitcast <8 x i64> [[TMP95]] to <16 x i32>
// CHECK:   [[TMP99:%.*]] = icmp ult <16 x i32> [[TMP94]], [[TMP96]]
// CHECK:   [[TMP100:%.*]] = select <16 x i1> [[TMP99]], <16 x i32> [[TMP94]], <16 x i32> [[TMP96]]
// CHECK:   [[TMP101:%.*]] = bitcast <16 x i32> [[TMP100]] to <8 x i64>
// CHECK:   store <8 x i64> [[TMP101]], <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP102:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK:   [[VECEXT_I:%.*]] = extractelement <8 x i64> [[TMP102]], i32 0
// CHECK:   [[CONV_I:%.*]] = trunc i64 [[VECEXT_I]] to i32
// CHECK:   ret i32 [[CONV_I]]
unsigned int test_mm512_mask_reduce_min_epu32(__mmask16 __M, __m512i __W){
  return _mm512_mask_reduce_min_epu32(__M, __W); 
}

// CHECK-LABEL: define float @test_mm512_mask_reduce_min_ps(i16 zeroext %__M, <16 x float> %__W) #0 {
// CHECK:   [[_COMPOUNDLITERAL_I_I18_I:%.*]] = alloca <16 x float>, align 64
// CHECK:   [[__A_ADDR_I19_I:%.*]] = alloca <16 x float>, align 64
// CHECK:   [[__B_ADDR_I20_I:%.*]] = alloca <16 x float>, align 64
// CHECK:   [[_COMPOUNDLITERAL_I_I15_I:%.*]] = alloca <16 x float>, align 64
// CHECK:   [[__A_ADDR_I16_I:%.*]] = alloca <16 x float>, align 64
// CHECK:   [[__B_ADDR_I17_I:%.*]] = alloca <16 x float>, align 64
// CHECK:   [[_COMPOUNDLITERAL_I_I12_I:%.*]] = alloca <16 x float>, align 64
// CHECK:   [[__A_ADDR_I13_I:%.*]] = alloca <16 x float>, align 64
// CHECK:   [[__B_ADDR_I14_I:%.*]] = alloca <16 x float>, align 64
// CHECK:   [[_COMPOUNDLITERAL_I_I_I:%.*]] = alloca <16 x float>, align 64
// CHECK:   [[__A_ADDR_I_I:%.*]] = alloca <16 x float>, align 64
// CHECK:   [[__B_ADDR_I_I:%.*]] = alloca <16 x float>, align 64
// CHECK:   [[__W_ADDR_I_I:%.*]] = alloca float, align 4
// CHECK:   [[_COMPOUNDLITERAL_I_I:%.*]] = alloca <16 x float>, align 64
// CHECK:   [[__M_ADDR_I:%.*]] = alloca i16, align 2
// CHECK:   [[__V_ADDR_I:%.*]] = alloca <16 x float>, align 64
// CHECK:   [[__M_ADDR:%.*]] = alloca i16, align 2
// CHECK:   [[__W_ADDR:%.*]] = alloca <16 x float>, align 64
// CHECK:   store i16 %__M, i16* [[__M_ADDR]], align 2
// CHECK:   store <16 x float> %__W, <16 x float>* [[__W_ADDR]], align 64
// CHECK:   [[TMP0:%.*]] = load i16, i16* [[__M_ADDR]], align 2
// CHECK:   [[TMP1:%.*]] = load <16 x float>, <16 x float>* [[__W_ADDR]], align 64
// CHECK:   store i16 [[TMP0]], i16* [[__M_ADDR_I]], align 2
// CHECK:   store <16 x float> [[TMP1]], <16 x float>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP2:%.*]] = load i16, i16* [[__M_ADDR_I]], align 2
// CHECK:   [[TMP3:%.*]] = load <16 x float>, <16 x float>* [[__V_ADDR_I]], align 64
// CHECK:   store float 0x7FF0000000000000, float* [[__W_ADDR_I_I]], align 4
// CHECK:   [[TMP4:%.*]] = load float, float* [[__W_ADDR_I_I]], align 4
// CHECK:   [[VECINIT_I_I:%.*]] = insertelement <16 x float> undef, float [[TMP4]], i32 0
// CHECK:   [[TMP5:%.*]] = load float, float* [[__W_ADDR_I_I]], align 4
// CHECK:   [[VECINIT1_I_I:%.*]] = insertelement <16 x float> [[VECINIT_I_I]], float [[TMP5]], i32 1
// CHECK:   [[TMP6:%.*]] = load float, float* [[__W_ADDR_I_I]], align 4
// CHECK:   [[VECINIT2_I_I:%.*]] = insertelement <16 x float> [[VECINIT1_I_I]], float [[TMP6]], i32 2
// CHECK:   [[TMP7:%.*]] = load float, float* [[__W_ADDR_I_I]], align 4
// CHECK:   [[VECINIT3_I_I:%.*]] = insertelement <16 x float> [[VECINIT2_I_I]], float [[TMP7]], i32 3
// CHECK:   [[TMP8:%.*]] = load float, float* [[__W_ADDR_I_I]], align 4
// CHECK:   [[VECINIT4_I_I:%.*]] = insertelement <16 x float> [[VECINIT3_I_I]], float [[TMP8]], i32 4
// CHECK:   [[TMP9:%.*]] = load float, float* [[__W_ADDR_I_I]], align 4
// CHECK:   [[VECINIT5_I_I:%.*]] = insertelement <16 x float> [[VECINIT4_I_I]], float [[TMP9]], i32 5
// CHECK:   [[TMP10:%.*]] = load float, float* [[__W_ADDR_I_I]], align 4
// CHECK:   [[VECINIT6_I_I:%.*]] = insertelement <16 x float> [[VECINIT5_I_I]], float [[TMP10]], i32 6
// CHECK:   [[TMP11:%.*]] = load float, float* [[__W_ADDR_I_I]], align 4
// CHECK:   [[VECINIT7_I_I:%.*]] = insertelement <16 x float> [[VECINIT6_I_I]], float [[TMP11]], i32 7
// CHECK:   [[TMP12:%.*]] = load float, float* [[__W_ADDR_I_I]], align 4
// CHECK:   [[VECINIT8_I_I:%.*]] = insertelement <16 x float> [[VECINIT7_I_I]], float [[TMP12]], i32 8
// CHECK:   [[TMP13:%.*]] = load float, float* [[__W_ADDR_I_I]], align 4
// CHECK:   [[VECINIT9_I_I:%.*]] = insertelement <16 x float> [[VECINIT8_I_I]], float [[TMP13]], i32 9
// CHECK:   [[TMP14:%.*]] = load float, float* [[__W_ADDR_I_I]], align 4
// CHECK:   [[VECINIT10_I_I:%.*]] = insertelement <16 x float> [[VECINIT9_I_I]], float [[TMP14]], i32 10
// CHECK:   [[TMP15:%.*]] = load float, float* [[__W_ADDR_I_I]], align 4
// CHECK:   [[VECINIT11_I_I:%.*]] = insertelement <16 x float> [[VECINIT10_I_I]], float [[TMP15]], i32 11
// CHECK:   [[TMP16:%.*]] = load float, float* [[__W_ADDR_I_I]], align 4
// CHECK:   [[VECINIT12_I_I:%.*]] = insertelement <16 x float> [[VECINIT11_I_I]], float [[TMP16]], i32 12
// CHECK:   [[TMP17:%.*]] = load float, float* [[__W_ADDR_I_I]], align 4
// CHECK:   [[VECINIT13_I_I:%.*]] = insertelement <16 x float> [[VECINIT12_I_I]], float [[TMP17]], i32 13
// CHECK:   [[TMP18:%.*]] = load float, float* [[__W_ADDR_I_I]], align 4
// CHECK:   [[VECINIT14_I_I:%.*]] = insertelement <16 x float> [[VECINIT13_I_I]], float [[TMP18]], i32 14
// CHECK:   [[TMP19:%.*]] = load float, float* [[__W_ADDR_I_I]], align 4
// CHECK:   [[VECINIT15_I_I:%.*]] = insertelement <16 x float> [[VECINIT14_I_I]], float [[TMP19]], i32 15
// CHECK:   store <16 x float> [[VECINIT15_I_I]], <16 x float>* [[_COMPOUNDLITERAL_I_I]], align 64
// CHECK:   [[TMP20:%.*]] = load <16 x float>, <16 x float>* [[_COMPOUNDLITERAL_I_I]], align 64
// CHECK:   [[TMP21:%.*]] = bitcast i16 [[TMP2]] to <16 x i1>
// CHECK:   [[TMP22:%.*]] = select <16 x i1> [[TMP21]], <16 x float> [[TMP3]], <16 x float> [[TMP20]]
// CHECK:   store <16 x float> [[TMP22]], <16 x float>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP23:%.*]] = load <16 x float>, <16 x float>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP24:%.*]] = load <16 x float>, <16 x float>* [[__V_ADDR_I]], align 64
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <16 x float> [[TMP23]], <16 x float> [[TMP24]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP25:%.*]] = load <16 x float>, <16 x float>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP26:%.*]] = load <16 x float>, <16 x float>* [[__V_ADDR_I]], align 64
// CHECK:   [[SHUFFLE1_I:%.*]] = shufflevector <16 x float> [[TMP25]], <16 x float> [[TMP26]], <16 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   store <16 x float> [[SHUFFLE_I]], <16 x float>* [[__A_ADDR_I19_I]], align 64
// CHECK:   store <16 x float> [[SHUFFLE1_I]], <16 x float>* [[__B_ADDR_I20_I]], align 64
// CHECK:   [[TMP27:%.*]] = load <16 x float>, <16 x float>* [[__A_ADDR_I19_I]], align 64
// CHECK:   [[TMP28:%.*]] = load <16 x float>, <16 x float>* [[__B_ADDR_I20_I]], align 64
// CHECK:   store <16 x float> zeroinitializer, <16 x float>* [[_COMPOUNDLITERAL_I_I18_I]], align 64
// CHECK:   [[TMP29:%.*]] = load <16 x float>, <16 x float>* [[_COMPOUNDLITERAL_I_I18_I]], align 64
// CHECK:   [[TMP30:%.*]] = call <16 x float> @llvm.x86.avx512.mask.min.ps.512(<16 x float> [[TMP27]], <16 x float> [[TMP28]], <16 x float> [[TMP29]], i16 -1, i32 4) #2
// CHECK:   store <16 x float> [[TMP30]], <16 x float>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP31:%.*]] = load <16 x float>, <16 x float>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP32:%.*]] = load <16 x float>, <16 x float>* [[__V_ADDR_I]], align 64
// CHECK:   [[SHUFFLE3_I:%.*]] = shufflevector <16 x float> [[TMP31]], <16 x float> [[TMP32]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP33:%.*]] = load <16 x float>, <16 x float>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP34:%.*]] = load <16 x float>, <16 x float>* [[__V_ADDR_I]], align 64
// CHECK:   [[SHUFFLE4_I:%.*]] = shufflevector <16 x float> [[TMP33]], <16 x float> [[TMP34]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   store <16 x float> [[SHUFFLE3_I]], <16 x float>* [[__A_ADDR_I16_I]], align 64
// CHECK:   store <16 x float> [[SHUFFLE4_I]], <16 x float>* [[__B_ADDR_I17_I]], align 64
// CHECK:   [[TMP35:%.*]] = load <16 x float>, <16 x float>* [[__A_ADDR_I16_I]], align 64
// CHECK:   [[TMP36:%.*]] = load <16 x float>, <16 x float>* [[__B_ADDR_I17_I]], align 64
// CHECK:   store <16 x float> zeroinitializer, <16 x float>* [[_COMPOUNDLITERAL_I_I15_I]], align 64
// CHECK:   [[TMP37:%.*]] = load <16 x float>, <16 x float>* [[_COMPOUNDLITERAL_I_I15_I]], align 64
// CHECK:   [[TMP38:%.*]] = call <16 x float> @llvm.x86.avx512.mask.min.ps.512(<16 x float> [[TMP35]], <16 x float> [[TMP36]], <16 x float> [[TMP37]], i16 -1, i32 4) #2
// CHECK:   store <16 x float> [[TMP38]], <16 x float>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP39:%.*]] = load <16 x float>, <16 x float>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP40:%.*]] = load <16 x float>, <16 x float>* [[__V_ADDR_I]], align 64
// CHECK:   [[SHUFFLE6_I:%.*]] = shufflevector <16 x float> [[TMP39]], <16 x float> [[TMP40]], <16 x i32> <i32 0, i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP41:%.*]] = load <16 x float>, <16 x float>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP42:%.*]] = load <16 x float>, <16 x float>* [[__V_ADDR_I]], align 64
// CHECK:   [[SHUFFLE7_I:%.*]] = shufflevector <16 x float> [[TMP41]], <16 x float> [[TMP42]], <16 x i32> <i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   store <16 x float> [[SHUFFLE6_I]], <16 x float>* [[__A_ADDR_I13_I]], align 64
// CHECK:   store <16 x float> [[SHUFFLE7_I]], <16 x float>* [[__B_ADDR_I14_I]], align 64
// CHECK:   [[TMP43:%.*]] = load <16 x float>, <16 x float>* [[__A_ADDR_I13_I]], align 64
// CHECK:   [[TMP44:%.*]] = load <16 x float>, <16 x float>* [[__B_ADDR_I14_I]], align 64
// CHECK:   store <16 x float> zeroinitializer, <16 x float>* [[_COMPOUNDLITERAL_I_I12_I]], align 64
// CHECK:   [[TMP45:%.*]] = load <16 x float>, <16 x float>* [[_COMPOUNDLITERAL_I_I12_I]], align 64
// CHECK:   [[TMP46:%.*]] = call <16 x float> @llvm.x86.avx512.mask.min.ps.512(<16 x float> [[TMP43]], <16 x float> [[TMP44]], <16 x float> [[TMP45]], i16 -1, i32 4) #2
// CHECK:   store <16 x float> [[TMP46]], <16 x float>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP47:%.*]] = load <16 x float>, <16 x float>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP48:%.*]] = load <16 x float>, <16 x float>* [[__V_ADDR_I]], align 64
// CHECK:   [[SHUFFLE9_I:%.*]] = shufflevector <16 x float> [[TMP47]], <16 x float> [[TMP48]], <16 x i32> <i32 0, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   [[TMP49:%.*]] = load <16 x float>, <16 x float>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP50:%.*]] = load <16 x float>, <16 x float>* [[__V_ADDR_I]], align 64
// CHECK:   [[SHUFFLE10_I:%.*]] = shufflevector <16 x float> [[TMP49]], <16 x float> [[TMP50]], <16 x i32> <i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
// CHECK:   store <16 x float> [[SHUFFLE9_I]], <16 x float>* [[__A_ADDR_I_I]], align 64
// CHECK:   store <16 x float> [[SHUFFLE10_I]], <16 x float>* [[__B_ADDR_I_I]], align 64
// CHECK:   [[TMP51:%.*]] = load <16 x float>, <16 x float>* [[__A_ADDR_I_I]], align 64
// CHECK:   [[TMP52:%.*]] = load <16 x float>, <16 x float>* [[__B_ADDR_I_I]], align 64
// CHECK:   store <16 x float> zeroinitializer, <16 x float>* [[_COMPOUNDLITERAL_I_I_I]], align 64
// CHECK:   [[TMP53:%.*]] = load <16 x float>, <16 x float>* [[_COMPOUNDLITERAL_I_I_I]], align 64
// CHECK:   [[TMP54:%.*]] = call <16 x float> @llvm.x86.avx512.mask.min.ps.512(<16 x float> [[TMP51]], <16 x float> [[TMP52]], <16 x float> [[TMP53]], i16 -1, i32 4) #2
// CHECK:   store <16 x float> [[TMP54]], <16 x float>* [[__V_ADDR_I]], align 64
// CHECK:   [[TMP55:%.*]] = load <16 x float>, <16 x float>* [[__V_ADDR_I]], align 64
// CHECK:   [[VECEXT_I:%.*]] = extractelement <16 x float> [[TMP55]], i32 0
// CHECK:   ret float [[VECEXT_I]]
float test_mm512_mask_reduce_min_ps(__mmask16 __M, __m512 __W){
  return _mm512_mask_reduce_min_ps(__M, __W); 
}

