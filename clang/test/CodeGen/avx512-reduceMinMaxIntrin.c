// RUN: %clang_cc1 -ffreestanding %s -O0 -triple=x86_64-apple-darwin -target-cpu skylake-avx512 -emit-llvm -o - -Wall -Werror | FileCheck %s

#include <immintrin.h>

// CHECK-LABEL: define i64 @test_mm512_reduce_max_epi64(<8 x i64> %__W) #0 {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[__A_ADDR_I7_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__B_ADDR_I8_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__A_ADDR_I5_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__B_ADDR_I6_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__A_ADDR_I_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__B_ADDR_I_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__V_ADDR_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__T1_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__T2_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__T3_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__T4_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__T5_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__T6_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__W_ADDR:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    store <8 x i64> [[__W:%.*]], <8 x i64>* [[__W_ADDR]], align 64
// CHECK-NEXT:    [[TMP0:%.*]] = load <8 x i64>, <8 x i64>* [[__W_ADDR]], align 64
// CHECK-NEXT:    store <8 x i64> [[TMP0]], <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK-NEXT:    [[TMP1:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK-NEXT:    [[TMP2:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK-NEXT:    [[SHUFFLE_I:%.*]] = shufflevector <8 x i64> [[TMP1]], <8 x i64> [[TMP2]], <8 x i32> <i32 4, i32 5, i32 6, i32 7, i32 0, i32 1, i32 2, i32 3>
// CHECK-NEXT:    store <8 x i64> [[SHUFFLE_I]], <8 x i64>* [[__T1_I]], align 64
// CHECK-NEXT:    [[TMP3:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK-NEXT:    [[TMP4:%.*]] = load <8 x i64>, <8 x i64>* [[__T1_I]], align 64
// CHECK-NEXT:    store <8 x i64> [[TMP3]], <8 x i64>* [[__A_ADDR_I_I]], align 64
// CHECK-NEXT:    store <8 x i64> [[TMP4]], <8 x i64>* [[__B_ADDR_I_I]], align 64
// CHECK-NEXT:    [[TMP5:%.*]] = load <8 x i64>, <8 x i64>* [[__A_ADDR_I_I]], align 64
// CHECK-NEXT:    [[TMP6:%.*]] = load <8 x i64>, <8 x i64>* [[__B_ADDR_I_I]], align 64
// CHECK-NEXT:    [[TMP7:%.*]] = icmp sgt <8 x i64> [[TMP5]], [[TMP6]]
// CHECK-NEXT:    [[TMP8:%.*]] = select <8 x i1> [[TMP7]], <8 x i64> [[TMP5]], <8 x i64> [[TMP6]]
// CHECK-NEXT:    store <8 x i64> [[TMP8]], <8 x i64>* [[__T2_I]], align 64
// CHECK-NEXT:    [[TMP9:%.*]] = load <8 x i64>, <8 x i64>* [[__T2_I]], align 64
// CHECK-NEXT:    [[TMP10:%.*]] = load <8 x i64>, <8 x i64>* [[__T2_I]], align 64
// CHECK-NEXT:    [[SHUFFLE1_I:%.*]] = shufflevector <8 x i64> [[TMP9]], <8 x i64> [[TMP10]], <8 x i32> <i32 2, i32 3, i32 0, i32 1, i32 6, i32 7, i32 4, i32 5>
// CHECK-NEXT:    store <8 x i64> [[SHUFFLE1_I]], <8 x i64>* [[__T3_I]], align 64
// CHECK-NEXT:    [[TMP11:%.*]] = load <8 x i64>, <8 x i64>* [[__T2_I]], align 64
// CHECK-NEXT:    [[TMP12:%.*]] = load <8 x i64>, <8 x i64>* [[__T3_I]], align 64
// CHECK-NEXT:    store <8 x i64> [[TMP11]], <8 x i64>* [[__A_ADDR_I7_I]], align 64
// CHECK-NEXT:    store <8 x i64> [[TMP12]], <8 x i64>* [[__B_ADDR_I8_I]], align 64
// CHECK-NEXT:    [[TMP13:%.*]] = load <8 x i64>, <8 x i64>* [[__A_ADDR_I7_I]], align 64
// CHECK-NEXT:    [[TMP14:%.*]] = load <8 x i64>, <8 x i64>* [[__B_ADDR_I8_I]], align 64
// CHECK-NEXT:    [[TMP15:%.*]] = icmp sgt <8 x i64> [[TMP13]], [[TMP14]]
// CHECK-NEXT:    [[TMP16:%.*]] = select <8 x i1> [[TMP15]], <8 x i64> [[TMP13]], <8 x i64> [[TMP14]]
// CHECK-NEXT:    store <8 x i64> [[TMP16]], <8 x i64>* [[__T4_I]], align 64
// CHECK-NEXT:    [[TMP17:%.*]] = load <8 x i64>, <8 x i64>* [[__T4_I]], align 64
// CHECK-NEXT:    [[TMP18:%.*]] = load <8 x i64>, <8 x i64>* [[__T4_I]], align 64
// CHECK-NEXT:    [[SHUFFLE3_I:%.*]] = shufflevector <8 x i64> [[TMP17]], <8 x i64> [[TMP18]], <8 x i32> <i32 1, i32 0, i32 3, i32 2, i32 5, i32 4, i32 7, i32 6>
// CHECK-NEXT:    store <8 x i64> [[SHUFFLE3_I]], <8 x i64>* [[__T5_I]], align 64
// CHECK-NEXT:    [[TMP19:%.*]] = load <8 x i64>, <8 x i64>* [[__T4_I]], align 64
// CHECK-NEXT:    [[TMP20:%.*]] = load <8 x i64>, <8 x i64>* [[__T5_I]], align 64
// CHECK-NEXT:    store <8 x i64> [[TMP19]], <8 x i64>* [[__A_ADDR_I5_I]], align 64
// CHECK-NEXT:    store <8 x i64> [[TMP20]], <8 x i64>* [[__B_ADDR_I6_I]], align 64
// CHECK-NEXT:    [[TMP21:%.*]] = load <8 x i64>, <8 x i64>* [[__A_ADDR_I5_I]], align 64
// CHECK-NEXT:    [[TMP22:%.*]] = load <8 x i64>, <8 x i64>* [[__B_ADDR_I6_I]], align 64
// CHECK-NEXT:    [[TMP23:%.*]] = icmp sgt <8 x i64> [[TMP21]], [[TMP22]]
// CHECK-NEXT:    [[TMP24:%.*]] = select <8 x i1> [[TMP23]], <8 x i64> [[TMP21]], <8 x i64> [[TMP22]]
// CHECK-NEXT:    store <8 x i64> [[TMP24]], <8 x i64>* [[__T6_I]], align 64
// CHECK-NEXT:    [[TMP25:%.*]] = load <8 x i64>, <8 x i64>* [[__T6_I]], align 64
// CHECK-NEXT:    [[VECEXT_I:%.*]] = extractelement <8 x i64> [[TMP25]], i32 0
// CHECK-NEXT:    ret i64 [[VECEXT_I]]
long long test_mm512_reduce_max_epi64(__m512i __W){
  return _mm512_reduce_max_epi64(__W);
}

// CHECK-LABEL: define i64 @test_mm512_reduce_max_epu64(<8 x i64> %__W) #0 {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[__A_ADDR_I7_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__B_ADDR_I8_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__A_ADDR_I5_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__B_ADDR_I6_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__A_ADDR_I_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__B_ADDR_I_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__V_ADDR_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__T1_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__T2_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__T3_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__T4_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__T5_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__T6_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__W_ADDR:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    store <8 x i64> [[__W:%.*]], <8 x i64>* [[__W_ADDR]], align 64
// CHECK-NEXT:    [[TMP0:%.*]] = load <8 x i64>, <8 x i64>* [[__W_ADDR]], align 64
// CHECK-NEXT:    store <8 x i64> [[TMP0]], <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK-NEXT:    [[TMP1:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK-NEXT:    [[TMP2:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK-NEXT:    [[SHUFFLE_I:%.*]] = shufflevector <8 x i64> [[TMP1]], <8 x i64> [[TMP2]], <8 x i32> <i32 4, i32 5, i32 6, i32 7, i32 0, i32 1, i32 2, i32 3>
// CHECK-NEXT:    store <8 x i64> [[SHUFFLE_I]], <8 x i64>* [[__T1_I]], align 64
// CHECK-NEXT:    [[TMP3:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK-NEXT:    [[TMP4:%.*]] = load <8 x i64>, <8 x i64>* [[__T1_I]], align 64
// CHECK-NEXT:    store <8 x i64> [[TMP3]], <8 x i64>* [[__A_ADDR_I_I]], align 64
// CHECK-NEXT:    store <8 x i64> [[TMP4]], <8 x i64>* [[__B_ADDR_I_I]], align 64
// CHECK-NEXT:    [[TMP5:%.*]] = load <8 x i64>, <8 x i64>* [[__A_ADDR_I_I]], align 64
// CHECK-NEXT:    [[TMP6:%.*]] = load <8 x i64>, <8 x i64>* [[__B_ADDR_I_I]], align 64
// CHECK-NEXT:    [[TMP7:%.*]] = icmp ugt <8 x i64> [[TMP5]], [[TMP6]]
// CHECK-NEXT:    [[TMP8:%.*]] = select <8 x i1> [[TMP7]], <8 x i64> [[TMP5]], <8 x i64> [[TMP6]]
// CHECK-NEXT:    store <8 x i64> [[TMP8]], <8 x i64>* [[__T2_I]], align 64
// CHECK-NEXT:    [[TMP9:%.*]] = load <8 x i64>, <8 x i64>* [[__T2_I]], align 64
// CHECK-NEXT:    [[TMP10:%.*]] = load <8 x i64>, <8 x i64>* [[__T2_I]], align 64
// CHECK-NEXT:    [[SHUFFLE1_I:%.*]] = shufflevector <8 x i64> [[TMP9]], <8 x i64> [[TMP10]], <8 x i32> <i32 2, i32 3, i32 0, i32 1, i32 6, i32 7, i32 4, i32 5>
// CHECK-NEXT:    store <8 x i64> [[SHUFFLE1_I]], <8 x i64>* [[__T3_I]], align 64
// CHECK-NEXT:    [[TMP11:%.*]] = load <8 x i64>, <8 x i64>* [[__T2_I]], align 64
// CHECK-NEXT:    [[TMP12:%.*]] = load <8 x i64>, <8 x i64>* [[__T3_I]], align 64
// CHECK-NEXT:    store <8 x i64> [[TMP11]], <8 x i64>* [[__A_ADDR_I7_I]], align 64
// CHECK-NEXT:    store <8 x i64> [[TMP12]], <8 x i64>* [[__B_ADDR_I8_I]], align 64
// CHECK-NEXT:    [[TMP13:%.*]] = load <8 x i64>, <8 x i64>* [[__A_ADDR_I7_I]], align 64
// CHECK-NEXT:    [[TMP14:%.*]] = load <8 x i64>, <8 x i64>* [[__B_ADDR_I8_I]], align 64
// CHECK-NEXT:    [[TMP15:%.*]] = icmp ugt <8 x i64> [[TMP13]], [[TMP14]]
// CHECK-NEXT:    [[TMP16:%.*]] = select <8 x i1> [[TMP15]], <8 x i64> [[TMP13]], <8 x i64> [[TMP14]]
// CHECK-NEXT:    store <8 x i64> [[TMP16]], <8 x i64>* [[__T4_I]], align 64
// CHECK-NEXT:    [[TMP17:%.*]] = load <8 x i64>, <8 x i64>* [[__T4_I]], align 64
// CHECK-NEXT:    [[TMP18:%.*]] = load <8 x i64>, <8 x i64>* [[__T4_I]], align 64
// CHECK-NEXT:    [[SHUFFLE3_I:%.*]] = shufflevector <8 x i64> [[TMP17]], <8 x i64> [[TMP18]], <8 x i32> <i32 1, i32 0, i32 3, i32 2, i32 5, i32 4, i32 7, i32 6>
// CHECK-NEXT:    store <8 x i64> [[SHUFFLE3_I]], <8 x i64>* [[__T5_I]], align 64
// CHECK-NEXT:    [[TMP19:%.*]] = load <8 x i64>, <8 x i64>* [[__T4_I]], align 64
// CHECK-NEXT:    [[TMP20:%.*]] = load <8 x i64>, <8 x i64>* [[__T5_I]], align 64
// CHECK-NEXT:    store <8 x i64> [[TMP19]], <8 x i64>* [[__A_ADDR_I5_I]], align 64
// CHECK-NEXT:    store <8 x i64> [[TMP20]], <8 x i64>* [[__B_ADDR_I6_I]], align 64
// CHECK-NEXT:    [[TMP21:%.*]] = load <8 x i64>, <8 x i64>* [[__A_ADDR_I5_I]], align 64
// CHECK-NEXT:    [[TMP22:%.*]] = load <8 x i64>, <8 x i64>* [[__B_ADDR_I6_I]], align 64
// CHECK-NEXT:    [[TMP23:%.*]] = icmp ugt <8 x i64> [[TMP21]], [[TMP22]]
// CHECK-NEXT:    [[TMP24:%.*]] = select <8 x i1> [[TMP23]], <8 x i64> [[TMP21]], <8 x i64> [[TMP22]]
// CHECK-NEXT:    store <8 x i64> [[TMP24]], <8 x i64>* [[__T6_I]], align 64
// CHECK-NEXT:    [[TMP25:%.*]] = load <8 x i64>, <8 x i64>* [[__T6_I]], align 64
// CHECK-NEXT:    [[VECEXT_I:%.*]] = extractelement <8 x i64> [[TMP25]], i32 0
// CHECK-NEXT:    ret i64 [[VECEXT_I]]
unsigned long long test_mm512_reduce_max_epu64(__m512i __W){
  return _mm512_reduce_max_epu64(__W); 
}

// CHECK-LABEL: define double @test_mm512_reduce_max_pd(<8 x double> %__W) #0 {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[__A_ADDR_I10_I:%.*]] = alloca <4 x double>, align 32
// CHECK-NEXT:    [[__B_ADDR_I11_I:%.*]] = alloca <4 x double>, align 32
// CHECK-NEXT:    [[__A_ADDR_I8_I:%.*]] = alloca <2 x double>, align 16
// CHECK-NEXT:    [[__B_ADDR_I9_I:%.*]] = alloca <2 x double>, align 16
// CHECK-NEXT:    [[__A_ADDR_I_I:%.*]] = alloca <2 x double>, align 16
// CHECK-NEXT:    [[__B_ADDR_I_I:%.*]] = alloca <2 x double>, align 16
// CHECK-NEXT:    [[__V_ADDR_I:%.*]] = alloca <8 x double>, align 64
// CHECK-NEXT:    [[__T1_I:%.*]] = alloca <4 x double>, align 32
// CHECK-NEXT:    [[__T2_I:%.*]] = alloca <4 x double>, align 32
// CHECK-NEXT:    [[__T3_I:%.*]] = alloca <4 x double>, align 32
// CHECK-NEXT:    [[__T4_I:%.*]] = alloca <2 x double>, align 16
// CHECK-NEXT:    [[__T5_I:%.*]] = alloca <2 x double>, align 16
// CHECK-NEXT:    [[__T6_I:%.*]] = alloca <2 x double>, align 16
// CHECK-NEXT:    [[__T7_I:%.*]] = alloca <2 x double>, align 16
// CHECK-NEXT:    [[__T8_I:%.*]] = alloca <2 x double>, align 16
// CHECK-NEXT:    [[__W_ADDR:%.*]] = alloca <8 x double>, align 64
// CHECK-NEXT:    store <8 x double> [[__W:%.*]], <8 x double>* [[__W_ADDR]], align 64
// CHECK-NEXT:    [[TMP0:%.*]] = load <8 x double>, <8 x double>* [[__W_ADDR]], align 64
// CHECK-NEXT:    store <8 x double> [[TMP0]], <8 x double>* [[__V_ADDR_I]], align 64
// CHECK-NEXT:    [[TMP1:%.*]] = load <8 x double>, <8 x double>* [[__V_ADDR_I]], align 64
// CHECK-NEXT:    [[EXTRACT_I:%.*]] = shufflevector <8 x double> [[TMP1]], <8 x double> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// CHECK-NEXT:    store <4 x double> [[EXTRACT_I]], <4 x double>* [[__T1_I]], align 32
// CHECK-NEXT:    [[TMP2:%.*]] = load <8 x double>, <8 x double>* [[__V_ADDR_I]], align 64
// CHECK-NEXT:    [[EXTRACT2_I:%.*]] = shufflevector <8 x double> [[TMP2]], <8 x double> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK-NEXT:    store <4 x double> [[EXTRACT2_I]], <4 x double>* [[__T2_I]], align 32
// CHECK-NEXT:    [[TMP3:%.*]] = load <4 x double>, <4 x double>* [[__T1_I]], align 32
// CHECK-NEXT:    [[TMP4:%.*]] = load <4 x double>, <4 x double>* [[__T2_I]], align 32
// CHECK-NEXT:    store <4 x double> [[TMP3]], <4 x double>* [[__A_ADDR_I10_I]], align 32
// CHECK-NEXT:    store <4 x double> [[TMP4]], <4 x double>* [[__B_ADDR_I11_I]], align 32
// CHECK-NEXT:    [[TMP5:%.*]] = load <4 x double>, <4 x double>* [[__A_ADDR_I10_I]], align 32
// CHECK-NEXT:    [[TMP6:%.*]] = load <4 x double>, <4 x double>* [[__B_ADDR_I11_I]], align 32
// CHECK-NEXT:    [[TMP7:%.*]] = call <4 x double> @llvm.x86.avx.max.pd.256(<4 x double> [[TMP5]], <4 x double> [[TMP6]]) #2
// CHECK-NEXT:    store <4 x double> [[TMP7]], <4 x double>* [[__T3_I]], align 32
// CHECK-NEXT:    [[TMP8:%.*]] = load <4 x double>, <4 x double>* [[__T3_I]], align 32
// CHECK-NEXT:    [[EXTRACT4_I:%.*]] = shufflevector <4 x double> [[TMP8]], <4 x double> undef, <2 x i32> <i32 0, i32 1>
// CHECK-NEXT:    store <2 x double> [[EXTRACT4_I]], <2 x double>* [[__T4_I]], align 16
// CHECK-NEXT:    [[TMP9:%.*]] = load <4 x double>, <4 x double>* [[__T3_I]], align 32
// CHECK-NEXT:    [[EXTRACT5_I:%.*]] = shufflevector <4 x double> [[TMP9]], <4 x double> undef, <2 x i32> <i32 2, i32 3>
// CHECK-NEXT:    store <2 x double> [[EXTRACT5_I]], <2 x double>* [[__T5_I]], align 16
// CHECK-NEXT:    [[TMP10:%.*]] = load <2 x double>, <2 x double>* [[__T4_I]], align 16
// CHECK-NEXT:    [[TMP11:%.*]] = load <2 x double>, <2 x double>* [[__T5_I]], align 16
// CHECK-NEXT:    store <2 x double> [[TMP10]], <2 x double>* [[__A_ADDR_I8_I]], align 16
// CHECK-NEXT:    store <2 x double> [[TMP11]], <2 x double>* [[__B_ADDR_I9_I]], align 16
// CHECK-NEXT:    [[TMP12:%.*]] = load <2 x double>, <2 x double>* [[__A_ADDR_I8_I]], align 16
// CHECK-NEXT:    [[TMP13:%.*]] = load <2 x double>, <2 x double>* [[__B_ADDR_I9_I]], align 16
// CHECK-NEXT:    [[TMP14:%.*]] = call <2 x double> @llvm.x86.sse2.max.pd(<2 x double> [[TMP12]], <2 x double> [[TMP13]]) #2
// CHECK-NEXT:    store <2 x double> [[TMP14]], <2 x double>* [[__T6_I]], align 16
// CHECK-NEXT:    [[TMP15:%.*]] = load <2 x double>, <2 x double>* [[__T6_I]], align 16
// CHECK-NEXT:    [[TMP16:%.*]] = load <2 x double>, <2 x double>* [[__T6_I]], align 16
// CHECK-NEXT:    [[SHUFFLE_I:%.*]] = shufflevector <2 x double> [[TMP15]], <2 x double> [[TMP16]], <2 x i32> <i32 1, i32 0>
// CHECK-NEXT:    store <2 x double> [[SHUFFLE_I]], <2 x double>* [[__T7_I]], align 16
// CHECK-NEXT:    [[TMP17:%.*]] = load <2 x double>, <2 x double>* [[__T6_I]], align 16
// CHECK-NEXT:    [[TMP18:%.*]] = load <2 x double>, <2 x double>* [[__T7_I]], align 16
// CHECK-NEXT:    store <2 x double> [[TMP17]], <2 x double>* [[__A_ADDR_I_I]], align 16
// CHECK-NEXT:    store <2 x double> [[TMP18]], <2 x double>* [[__B_ADDR_I_I]], align 16
// CHECK-NEXT:    [[TMP19:%.*]] = load <2 x double>, <2 x double>* [[__A_ADDR_I_I]], align 16
// CHECK-NEXT:    [[TMP20:%.*]] = load <2 x double>, <2 x double>* [[__B_ADDR_I_I]], align 16
// CHECK-NEXT:    [[TMP21:%.*]] = call <2 x double> @llvm.x86.sse2.max.pd(<2 x double> [[TMP19]], <2 x double> [[TMP20]]) #2
// CHECK-NEXT:    store <2 x double> [[TMP21]], <2 x double>* [[__T8_I]], align 16
// CHECK-NEXT:    [[TMP22:%.*]] = load <2 x double>, <2 x double>* [[__T8_I]], align 16
// CHECK-NEXT:    [[VECEXT_I:%.*]] = extractelement <2 x double> [[TMP22]], i32 0
// CHECK-NEXT:    ret double [[VECEXT_I]]
double test_mm512_reduce_max_pd(__m512d __W){
  return _mm512_reduce_max_pd(__W); 
}

// CHECK-LABEL: define i64 @test_mm512_reduce_min_epi64(<8 x i64> %__W) #0 {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[__A_ADDR_I7_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__B_ADDR_I8_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__A_ADDR_I5_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__B_ADDR_I6_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__A_ADDR_I_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__B_ADDR_I_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__V_ADDR_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__T1_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__T2_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__T3_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__T4_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__T5_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__T6_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__W_ADDR:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    store <8 x i64> [[__W:%.*]], <8 x i64>* [[__W_ADDR]], align 64
// CHECK-NEXT:    [[TMP0:%.*]] = load <8 x i64>, <8 x i64>* [[__W_ADDR]], align 64
// CHECK-NEXT:    store <8 x i64> [[TMP0]], <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK-NEXT:    [[TMP1:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK-NEXT:    [[TMP2:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK-NEXT:    [[SHUFFLE_I:%.*]] = shufflevector <8 x i64> [[TMP1]], <8 x i64> [[TMP2]], <8 x i32> <i32 4, i32 5, i32 6, i32 7, i32 0, i32 1, i32 2, i32 3>
// CHECK-NEXT:    store <8 x i64> [[SHUFFLE_I]], <8 x i64>* [[__T1_I]], align 64
// CHECK-NEXT:    [[TMP3:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK-NEXT:    [[TMP4:%.*]] = load <8 x i64>, <8 x i64>* [[__T1_I]], align 64
// CHECK-NEXT:    store <8 x i64> [[TMP3]], <8 x i64>* [[__A_ADDR_I_I]], align 64
// CHECK-NEXT:    store <8 x i64> [[TMP4]], <8 x i64>* [[__B_ADDR_I_I]], align 64
// CHECK-NEXT:    [[TMP5:%.*]] = load <8 x i64>, <8 x i64>* [[__A_ADDR_I_I]], align 64
// CHECK-NEXT:    [[TMP6:%.*]] = load <8 x i64>, <8 x i64>* [[__B_ADDR_I_I]], align 64
// CHECK-NEXT:    [[TMP7:%.*]] = icmp slt <8 x i64> [[TMP5]], [[TMP6]]
// CHECK-NEXT:    [[TMP8:%.*]] = select <8 x i1> [[TMP7]], <8 x i64> [[TMP5]], <8 x i64> [[TMP6]]
// CHECK-NEXT:    store <8 x i64> [[TMP8]], <8 x i64>* [[__T2_I]], align 64
// CHECK-NEXT:    [[TMP9:%.*]] = load <8 x i64>, <8 x i64>* [[__T2_I]], align 64
// CHECK-NEXT:    [[TMP10:%.*]] = load <8 x i64>, <8 x i64>* [[__T2_I]], align 64
// CHECK-NEXT:    [[SHUFFLE1_I:%.*]] = shufflevector <8 x i64> [[TMP9]], <8 x i64> [[TMP10]], <8 x i32> <i32 2, i32 3, i32 0, i32 1, i32 6, i32 7, i32 4, i32 5>
// CHECK-NEXT:    store <8 x i64> [[SHUFFLE1_I]], <8 x i64>* [[__T3_I]], align 64
// CHECK-NEXT:    [[TMP11:%.*]] = load <8 x i64>, <8 x i64>* [[__T2_I]], align 64
// CHECK-NEXT:    [[TMP12:%.*]] = load <8 x i64>, <8 x i64>* [[__T3_I]], align 64
// CHECK-NEXT:    store <8 x i64> [[TMP11]], <8 x i64>* [[__A_ADDR_I7_I]], align 64
// CHECK-NEXT:    store <8 x i64> [[TMP12]], <8 x i64>* [[__B_ADDR_I8_I]], align 64
// CHECK-NEXT:    [[TMP13:%.*]] = load <8 x i64>, <8 x i64>* [[__A_ADDR_I7_I]], align 64
// CHECK-NEXT:    [[TMP14:%.*]] = load <8 x i64>, <8 x i64>* [[__B_ADDR_I8_I]], align 64
// CHECK-NEXT:    [[TMP15:%.*]] = icmp slt <8 x i64> [[TMP13]], [[TMP14]]
// CHECK-NEXT:    [[TMP16:%.*]] = select <8 x i1> [[TMP15]], <8 x i64> [[TMP13]], <8 x i64> [[TMP14]]
// CHECK-NEXT:    store <8 x i64> [[TMP16]], <8 x i64>* [[__T4_I]], align 64
// CHECK-NEXT:    [[TMP17:%.*]] = load <8 x i64>, <8 x i64>* [[__T4_I]], align 64
// CHECK-NEXT:    [[TMP18:%.*]] = load <8 x i64>, <8 x i64>* [[__T4_I]], align 64
// CHECK-NEXT:    [[SHUFFLE3_I:%.*]] = shufflevector <8 x i64> [[TMP17]], <8 x i64> [[TMP18]], <8 x i32> <i32 1, i32 0, i32 3, i32 2, i32 5, i32 4, i32 7, i32 6>
// CHECK-NEXT:    store <8 x i64> [[SHUFFLE3_I]], <8 x i64>* [[__T5_I]], align 64
// CHECK-NEXT:    [[TMP19:%.*]] = load <8 x i64>, <8 x i64>* [[__T4_I]], align 64
// CHECK-NEXT:    [[TMP20:%.*]] = load <8 x i64>, <8 x i64>* [[__T5_I]], align 64
// CHECK-NEXT:    store <8 x i64> [[TMP19]], <8 x i64>* [[__A_ADDR_I5_I]], align 64
// CHECK-NEXT:    store <8 x i64> [[TMP20]], <8 x i64>* [[__B_ADDR_I6_I]], align 64
// CHECK-NEXT:    [[TMP21:%.*]] = load <8 x i64>, <8 x i64>* [[__A_ADDR_I5_I]], align 64
// CHECK-NEXT:    [[TMP22:%.*]] = load <8 x i64>, <8 x i64>* [[__B_ADDR_I6_I]], align 64
// CHECK-NEXT:    [[TMP23:%.*]] = icmp slt <8 x i64> [[TMP21]], [[TMP22]]
// CHECK-NEXT:    [[TMP24:%.*]] = select <8 x i1> [[TMP23]], <8 x i64> [[TMP21]], <8 x i64> [[TMP22]]
// CHECK-NEXT:    store <8 x i64> [[TMP24]], <8 x i64>* [[__T6_I]], align 64
// CHECK-NEXT:    [[TMP25:%.*]] = load <8 x i64>, <8 x i64>* [[__T6_I]], align 64
// CHECK-NEXT:    [[VECEXT_I:%.*]] = extractelement <8 x i64> [[TMP25]], i32 0
// CHECK-NEXT:    ret i64 [[VECEXT_I]]
long long test_mm512_reduce_min_epi64(__m512i __W){
  return _mm512_reduce_min_epi64(__W);
}

// CHECK-LABEL: define i64 @test_mm512_reduce_min_epu64(<8 x i64> %__W) #0 {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[__A_ADDR_I7_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__B_ADDR_I8_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__A_ADDR_I5_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__B_ADDR_I6_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__A_ADDR_I_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__B_ADDR_I_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__V_ADDR_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__T1_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__T2_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__T3_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__T4_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__T5_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__T6_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__W_ADDR:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    store <8 x i64> [[__W:%.*]], <8 x i64>* [[__W_ADDR]], align 64
// CHECK-NEXT:    [[TMP0:%.*]] = load <8 x i64>, <8 x i64>* [[__W_ADDR]], align 64
// CHECK-NEXT:    store <8 x i64> [[TMP0]], <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK-NEXT:    [[TMP1:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK-NEXT:    [[TMP2:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK-NEXT:    [[SHUFFLE_I:%.*]] = shufflevector <8 x i64> [[TMP1]], <8 x i64> [[TMP2]], <8 x i32> <i32 4, i32 5, i32 6, i32 7, i32 0, i32 1, i32 2, i32 3>
// CHECK-NEXT:    store <8 x i64> [[SHUFFLE_I]], <8 x i64>* [[__T1_I]], align 64
// CHECK-NEXT:    [[TMP3:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK-NEXT:    [[TMP4:%.*]] = load <8 x i64>, <8 x i64>* [[__T1_I]], align 64
// CHECK-NEXT:    store <8 x i64> [[TMP3]], <8 x i64>* [[__A_ADDR_I_I]], align 64
// CHECK-NEXT:    store <8 x i64> [[TMP4]], <8 x i64>* [[__B_ADDR_I_I]], align 64
// CHECK-NEXT:    [[TMP5:%.*]] = load <8 x i64>, <8 x i64>* [[__A_ADDR_I_I]], align 64
// CHECK-NEXT:    [[TMP6:%.*]] = load <8 x i64>, <8 x i64>* [[__B_ADDR_I_I]], align 64
// CHECK-NEXT:    [[TMP7:%.*]] = icmp ult <8 x i64> [[TMP5]], [[TMP6]]
// CHECK-NEXT:    [[TMP8:%.*]] = select <8 x i1> [[TMP7]], <8 x i64> [[TMP5]], <8 x i64> [[TMP6]]
// CHECK-NEXT:    store <8 x i64> [[TMP8]], <8 x i64>* [[__T2_I]], align 64
// CHECK-NEXT:    [[TMP9:%.*]] = load <8 x i64>, <8 x i64>* [[__T2_I]], align 64
// CHECK-NEXT:    [[TMP10:%.*]] = load <8 x i64>, <8 x i64>* [[__T2_I]], align 64
// CHECK-NEXT:    [[SHUFFLE1_I:%.*]] = shufflevector <8 x i64> [[TMP9]], <8 x i64> [[TMP10]], <8 x i32> <i32 2, i32 3, i32 0, i32 1, i32 6, i32 7, i32 4, i32 5>
// CHECK-NEXT:    store <8 x i64> [[SHUFFLE1_I]], <8 x i64>* [[__T3_I]], align 64
// CHECK-NEXT:    [[TMP11:%.*]] = load <8 x i64>, <8 x i64>* [[__T2_I]], align 64
// CHECK-NEXT:    [[TMP12:%.*]] = load <8 x i64>, <8 x i64>* [[__T3_I]], align 64
// CHECK-NEXT:    store <8 x i64> [[TMP11]], <8 x i64>* [[__A_ADDR_I7_I]], align 64
// CHECK-NEXT:    store <8 x i64> [[TMP12]], <8 x i64>* [[__B_ADDR_I8_I]], align 64
// CHECK-NEXT:    [[TMP13:%.*]] = load <8 x i64>, <8 x i64>* [[__A_ADDR_I7_I]], align 64
// CHECK-NEXT:    [[TMP14:%.*]] = load <8 x i64>, <8 x i64>* [[__B_ADDR_I8_I]], align 64
// CHECK-NEXT:    [[TMP15:%.*]] = icmp ult <8 x i64> [[TMP13]], [[TMP14]]
// CHECK-NEXT:    [[TMP16:%.*]] = select <8 x i1> [[TMP15]], <8 x i64> [[TMP13]], <8 x i64> [[TMP14]]
// CHECK-NEXT:    store <8 x i64> [[TMP16]], <8 x i64>* [[__T4_I]], align 64
// CHECK-NEXT:    [[TMP17:%.*]] = load <8 x i64>, <8 x i64>* [[__T4_I]], align 64
// CHECK-NEXT:    [[TMP18:%.*]] = load <8 x i64>, <8 x i64>* [[__T4_I]], align 64
// CHECK-NEXT:    [[SHUFFLE3_I:%.*]] = shufflevector <8 x i64> [[TMP17]], <8 x i64> [[TMP18]], <8 x i32> <i32 1, i32 0, i32 3, i32 2, i32 5, i32 4, i32 7, i32 6>
// CHECK-NEXT:    store <8 x i64> [[SHUFFLE3_I]], <8 x i64>* [[__T5_I]], align 64
// CHECK-NEXT:    [[TMP19:%.*]] = load <8 x i64>, <8 x i64>* [[__T4_I]], align 64
// CHECK-NEXT:    [[TMP20:%.*]] = load <8 x i64>, <8 x i64>* [[__T5_I]], align 64
// CHECK-NEXT:    store <8 x i64> [[TMP19]], <8 x i64>* [[__A_ADDR_I5_I]], align 64
// CHECK-NEXT:    store <8 x i64> [[TMP20]], <8 x i64>* [[__B_ADDR_I6_I]], align 64
// CHECK-NEXT:    [[TMP21:%.*]] = load <8 x i64>, <8 x i64>* [[__A_ADDR_I5_I]], align 64
// CHECK-NEXT:    [[TMP22:%.*]] = load <8 x i64>, <8 x i64>* [[__B_ADDR_I6_I]], align 64
// CHECK-NEXT:    [[TMP23:%.*]] = icmp ult <8 x i64> [[TMP21]], [[TMP22]]
// CHECK-NEXT:    [[TMP24:%.*]] = select <8 x i1> [[TMP23]], <8 x i64> [[TMP21]], <8 x i64> [[TMP22]]
// CHECK-NEXT:    store <8 x i64> [[TMP24]], <8 x i64>* [[__T6_I]], align 64
// CHECK-NEXT:    [[TMP25:%.*]] = load <8 x i64>, <8 x i64>* [[__T6_I]], align 64
// CHECK-NEXT:    [[VECEXT_I:%.*]] = extractelement <8 x i64> [[TMP25]], i32 0
// CHECK-NEXT:    ret i64 [[VECEXT_I]]
unsigned long long test_mm512_reduce_min_epu64(__m512i __W){
  return _mm512_reduce_min_epu64(__W);
}

// CHECK-LABEL: define double @test_mm512_reduce_min_pd(<8 x double> %__W) #0 {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[__A_ADDR_I10_I:%.*]] = alloca <4 x double>, align 32
// CHECK-NEXT:    [[__B_ADDR_I11_I:%.*]] = alloca <4 x double>, align 32
// CHECK-NEXT:    [[__A_ADDR_I8_I:%.*]] = alloca <2 x double>, align 16
// CHECK-NEXT:    [[__B_ADDR_I9_I:%.*]] = alloca <2 x double>, align 16
// CHECK-NEXT:    [[__A_ADDR_I_I:%.*]] = alloca <2 x double>, align 16
// CHECK-NEXT:    [[__B_ADDR_I_I:%.*]] = alloca <2 x double>, align 16
// CHECK-NEXT:    [[__V_ADDR_I:%.*]] = alloca <8 x double>, align 64
// CHECK-NEXT:    [[__T1_I:%.*]] = alloca <4 x double>, align 32
// CHECK-NEXT:    [[__T2_I:%.*]] = alloca <4 x double>, align 32
// CHECK-NEXT:    [[__T3_I:%.*]] = alloca <4 x double>, align 32
// CHECK-NEXT:    [[__T4_I:%.*]] = alloca <2 x double>, align 16
// CHECK-NEXT:    [[__T5_I:%.*]] = alloca <2 x double>, align 16
// CHECK-NEXT:    [[__T6_I:%.*]] = alloca <2 x double>, align 16
// CHECK-NEXT:    [[__T7_I:%.*]] = alloca <2 x double>, align 16
// CHECK-NEXT:    [[__T8_I:%.*]] = alloca <2 x double>, align 16
// CHECK-NEXT:    [[__W_ADDR:%.*]] = alloca <8 x double>, align 64
// CHECK-NEXT:    store <8 x double> [[__W:%.*]], <8 x double>* [[__W_ADDR]], align 64
// CHECK-NEXT:    [[TMP0:%.*]] = load <8 x double>, <8 x double>* [[__W_ADDR]], align 64
// CHECK-NEXT:    store <8 x double> [[TMP0]], <8 x double>* [[__V_ADDR_I]], align 64
// CHECK-NEXT:    [[TMP1:%.*]] = load <8 x double>, <8 x double>* [[__V_ADDR_I]], align 64
// CHECK-NEXT:    [[EXTRACT_I:%.*]] = shufflevector <8 x double> [[TMP1]], <8 x double> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// CHECK-NEXT:    store <4 x double> [[EXTRACT_I]], <4 x double>* [[__T1_I]], align 32
// CHECK-NEXT:    [[TMP2:%.*]] = load <8 x double>, <8 x double>* [[__V_ADDR_I]], align 64
// CHECK-NEXT:    [[EXTRACT2_I:%.*]] = shufflevector <8 x double> [[TMP2]], <8 x double> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK-NEXT:    store <4 x double> [[EXTRACT2_I]], <4 x double>* [[__T2_I]], align 32
// CHECK-NEXT:    [[TMP3:%.*]] = load <4 x double>, <4 x double>* [[__T1_I]], align 32
// CHECK-NEXT:    [[TMP4:%.*]] = load <4 x double>, <4 x double>* [[__T2_I]], align 32
// CHECK-NEXT:    store <4 x double> [[TMP3]], <4 x double>* [[__A_ADDR_I10_I]], align 32
// CHECK-NEXT:    store <4 x double> [[TMP4]], <4 x double>* [[__B_ADDR_I11_I]], align 32
// CHECK-NEXT:    [[TMP5:%.*]] = load <4 x double>, <4 x double>* [[__A_ADDR_I10_I]], align 32
// CHECK-NEXT:    [[TMP6:%.*]] = load <4 x double>, <4 x double>* [[__B_ADDR_I11_I]], align 32
// CHECK-NEXT:    [[TMP7:%.*]] = call <4 x double> @llvm.x86.avx.min.pd.256(<4 x double> [[TMP5]], <4 x double> [[TMP6]]) #2
// CHECK-NEXT:    store <4 x double> [[TMP7]], <4 x double>* [[__T3_I]], align 32
// CHECK-NEXT:    [[TMP8:%.*]] = load <4 x double>, <4 x double>* [[__T3_I]], align 32
// CHECK-NEXT:    [[EXTRACT4_I:%.*]] = shufflevector <4 x double> [[TMP8]], <4 x double> undef, <2 x i32> <i32 0, i32 1>
// CHECK-NEXT:    store <2 x double> [[EXTRACT4_I]], <2 x double>* [[__T4_I]], align 16
// CHECK-NEXT:    [[TMP9:%.*]] = load <4 x double>, <4 x double>* [[__T3_I]], align 32
// CHECK-NEXT:    [[EXTRACT5_I:%.*]] = shufflevector <4 x double> [[TMP9]], <4 x double> undef, <2 x i32> <i32 2, i32 3>
// CHECK-NEXT:    store <2 x double> [[EXTRACT5_I]], <2 x double>* [[__T5_I]], align 16
// CHECK-NEXT:    [[TMP10:%.*]] = load <2 x double>, <2 x double>* [[__T4_I]], align 16
// CHECK-NEXT:    [[TMP11:%.*]] = load <2 x double>, <2 x double>* [[__T5_I]], align 16
// CHECK-NEXT:    store <2 x double> [[TMP10]], <2 x double>* [[__A_ADDR_I8_I]], align 16
// CHECK-NEXT:    store <2 x double> [[TMP11]], <2 x double>* [[__B_ADDR_I9_I]], align 16
// CHECK-NEXT:    [[TMP12:%.*]] = load <2 x double>, <2 x double>* [[__A_ADDR_I8_I]], align 16
// CHECK-NEXT:    [[TMP13:%.*]] = load <2 x double>, <2 x double>* [[__B_ADDR_I9_I]], align 16
// CHECK-NEXT:    [[TMP14:%.*]] = call <2 x double> @llvm.x86.sse2.min.pd(<2 x double> [[TMP12]], <2 x double> [[TMP13]]) #2
// CHECK-NEXT:    store <2 x double> [[TMP14]], <2 x double>* [[__T6_I]], align 16
// CHECK-NEXT:    [[TMP15:%.*]] = load <2 x double>, <2 x double>* [[__T6_I]], align 16
// CHECK-NEXT:    [[TMP16:%.*]] = load <2 x double>, <2 x double>* [[__T6_I]], align 16
// CHECK-NEXT:    [[SHUFFLE_I:%.*]] = shufflevector <2 x double> [[TMP15]], <2 x double> [[TMP16]], <2 x i32> <i32 1, i32 0>
// CHECK-NEXT:    store <2 x double> [[SHUFFLE_I]], <2 x double>* [[__T7_I]], align 16
// CHECK-NEXT:    [[TMP17:%.*]] = load <2 x double>, <2 x double>* [[__T6_I]], align 16
// CHECK-NEXT:    [[TMP18:%.*]] = load <2 x double>, <2 x double>* [[__T7_I]], align 16
// CHECK-NEXT:    store <2 x double> [[TMP17]], <2 x double>* [[__A_ADDR_I_I]], align 16
// CHECK-NEXT:    store <2 x double> [[TMP18]], <2 x double>* [[__B_ADDR_I_I]], align 16
// CHECK-NEXT:    [[TMP19:%.*]] = load <2 x double>, <2 x double>* [[__A_ADDR_I_I]], align 16
// CHECK-NEXT:    [[TMP20:%.*]] = load <2 x double>, <2 x double>* [[__B_ADDR_I_I]], align 16
// CHECK-NEXT:    [[TMP21:%.*]] = call <2 x double> @llvm.x86.sse2.min.pd(<2 x double> [[TMP19]], <2 x double> [[TMP20]]) #2
// CHECK-NEXT:    store <2 x double> [[TMP21]], <2 x double>* [[__T8_I]], align 16
// CHECK-NEXT:    [[TMP22:%.*]] = load <2 x double>, <2 x double>* [[__T8_I]], align 16
// CHECK-NEXT:    [[VECEXT_I:%.*]] = extractelement <2 x double> [[TMP22]], i32 0
// CHECK-NEXT:    ret double [[VECEXT_I]]
double test_mm512_reduce_min_pd(__m512d __W){
  return _mm512_reduce_min_pd(__W); 
}

// CHECK-LABEL: define i64 @test_mm512_mask_reduce_max_epi64(i8 zeroext %__M, <8 x i64> %__W) #0 {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[__W_ADDR_I_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__U_ADDR_I_I:%.*]] = alloca i8, align 1
// CHECK-NEXT:    [[__A_ADDR_I11_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__A_ADDR_I9_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__B_ADDR_I10_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__A_ADDR_I7_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__B_ADDR_I8_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__A_ADDR_I_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__B_ADDR_I_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__D_ADDR_I_I:%.*]] = alloca i64, align 8
// CHECK-NEXT:    [[DOTCOMPOUNDLITERAL_I_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__M_ADDR_I:%.*]] = alloca i8, align 1
// CHECK-NEXT:    [[__V_ADDR_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__T1_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__T2_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__T3_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__T4_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__T5_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__T6_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__M_ADDR:%.*]] = alloca i8, align 1
// CHECK-NEXT:    [[__W_ADDR:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    store i8 [[__M:%.*]], i8* [[__M_ADDR]], align 1
// CHECK-NEXT:    store <8 x i64> [[__W:%.*]], <8 x i64>* [[__W_ADDR]], align 64
// CHECK-NEXT:    [[TMP0:%.*]] = load i8, i8* [[__M_ADDR]], align 1
// CHECK-NEXT:    [[TMP1:%.*]] = load <8 x i64>, <8 x i64>* [[__W_ADDR]], align 64
// CHECK-NEXT:    store i8 [[TMP0]], i8* [[__M_ADDR_I]], align 1
// CHECK-NEXT:    store <8 x i64> [[TMP1]], <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK-NEXT:    store i64 -9223372036854775808, i64* [[__D_ADDR_I_I]], align 8
// CHECK-NEXT:    [[TMP2:%.*]] = load i64, i64* [[__D_ADDR_I_I]], align 8
// CHECK-NEXT:    [[VECINIT_I_I:%.*]] = insertelement <8 x i64> undef, i64 [[TMP2]], i32 0
// CHECK-NEXT:    [[TMP3:%.*]] = load i64, i64* [[__D_ADDR_I_I]], align 8
// CHECK-NEXT:    [[VECINIT1_I_I:%.*]] = insertelement <8 x i64> [[VECINIT_I_I]], i64 [[TMP3]], i32 1
// CHECK-NEXT:    [[TMP4:%.*]] = load i64, i64* [[__D_ADDR_I_I]], align 8
// CHECK-NEXT:    [[VECINIT2_I_I:%.*]] = insertelement <8 x i64> [[VECINIT1_I_I]], i64 [[TMP4]], i32 2
// CHECK-NEXT:    [[TMP5:%.*]] = load i64, i64* [[__D_ADDR_I_I]], align 8
// CHECK-NEXT:    [[VECINIT3_I_I:%.*]] = insertelement <8 x i64> [[VECINIT2_I_I]], i64 [[TMP5]], i32 3
// CHECK-NEXT:    [[TMP6:%.*]] = load i64, i64* [[__D_ADDR_I_I]], align 8
// CHECK-NEXT:    [[VECINIT4_I_I:%.*]] = insertelement <8 x i64> [[VECINIT3_I_I]], i64 [[TMP6]], i32 4
// CHECK-NEXT:    [[TMP7:%.*]] = load i64, i64* [[__D_ADDR_I_I]], align 8
// CHECK-NEXT:    [[VECINIT5_I_I:%.*]] = insertelement <8 x i64> [[VECINIT4_I_I]], i64 [[TMP7]], i32 5
// CHECK-NEXT:    [[TMP8:%.*]] = load i64, i64* [[__D_ADDR_I_I]], align 8
// CHECK-NEXT:    [[VECINIT6_I_I:%.*]] = insertelement <8 x i64> [[VECINIT5_I_I]], i64 [[TMP8]], i32 6
// CHECK-NEXT:    [[TMP9:%.*]] = load i64, i64* [[__D_ADDR_I_I]], align 8
// CHECK-NEXT:    [[VECINIT7_I_I:%.*]] = insertelement <8 x i64> [[VECINIT6_I_I]], i64 [[TMP9]], i32 7
// CHECK-NEXT:    store <8 x i64> [[VECINIT7_I_I]], <8 x i64>* [[DOTCOMPOUNDLITERAL_I_I]], align 64
// CHECK-NEXT:    [[TMP10:%.*]] = load <8 x i64>, <8 x i64>* [[DOTCOMPOUNDLITERAL_I_I]], align 64
// CHECK-NEXT:    [[TMP11:%.*]] = load i8, i8* [[__M_ADDR_I]], align 1
// CHECK-NEXT:    [[TMP12:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK-NEXT:    store <8 x i64> [[TMP10]], <8 x i64>* [[__W_ADDR_I_I]], align 64
// CHECK-NEXT:    store i8 [[TMP11]], i8* [[__U_ADDR_I_I]], align 1
// CHECK-NEXT:    store <8 x i64> [[TMP12]], <8 x i64>* [[__A_ADDR_I11_I]], align 64
// CHECK-NEXT:    [[TMP13:%.*]] = load i8, i8* [[__U_ADDR_I_I]], align 1
// CHECK-NEXT:    [[TMP14:%.*]] = load <8 x i64>, <8 x i64>* [[__A_ADDR_I11_I]], align 64
// CHECK-NEXT:    [[TMP15:%.*]] = load <8 x i64>, <8 x i64>* [[__W_ADDR_I_I]], align 64
// CHECK-NEXT:    [[TMP16:%.*]] = bitcast i8 [[TMP13]] to <8 x i1>
// CHECK-NEXT:    [[TMP17:%.*]] = select <8 x i1> [[TMP16]], <8 x i64> [[TMP14]], <8 x i64> [[TMP15]]
// CHECK-NEXT:    store <8 x i64> [[TMP17]], <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK-NEXT:    [[TMP18:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK-NEXT:    [[TMP19:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK-NEXT:    [[SHUFFLE_I:%.*]] = shufflevector <8 x i64> [[TMP18]], <8 x i64> [[TMP19]], <8 x i32> <i32 4, i32 5, i32 6, i32 7, i32 0, i32 1, i32 2, i32 3>
// CHECK-NEXT:    store <8 x i64> [[SHUFFLE_I]], <8 x i64>* [[__T1_I]], align 64
// CHECK-NEXT:    [[TMP20:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK-NEXT:    [[TMP21:%.*]] = load <8 x i64>, <8 x i64>* [[__T1_I]], align 64
// CHECK-NEXT:    store <8 x i64> [[TMP20]], <8 x i64>* [[__A_ADDR_I9_I]], align 64
// CHECK-NEXT:    store <8 x i64> [[TMP21]], <8 x i64>* [[__B_ADDR_I10_I]], align 64
// CHECK-NEXT:    [[TMP22:%.*]] = load <8 x i64>, <8 x i64>* [[__A_ADDR_I9_I]], align 64
// CHECK-NEXT:    [[TMP23:%.*]] = load <8 x i64>, <8 x i64>* [[__B_ADDR_I10_I]], align 64
// CHECK-NEXT:    [[TMP24:%.*]] = icmp sgt <8 x i64> [[TMP22]], [[TMP23]]
// CHECK-NEXT:    [[TMP25:%.*]] = select <8 x i1> [[TMP24]], <8 x i64> [[TMP22]], <8 x i64> [[TMP23]]
// CHECK-NEXT:    store <8 x i64> [[TMP25]], <8 x i64>* [[__T2_I]], align 64
// CHECK-NEXT:    [[TMP26:%.*]] = load <8 x i64>, <8 x i64>* [[__T2_I]], align 64
// CHECK-NEXT:    [[TMP27:%.*]] = load <8 x i64>, <8 x i64>* [[__T2_I]], align 64
// CHECK-NEXT:    [[SHUFFLE3_I:%.*]] = shufflevector <8 x i64> [[TMP26]], <8 x i64> [[TMP27]], <8 x i32> <i32 2, i32 3, i32 0, i32 1, i32 6, i32 7, i32 4, i32 5>
// CHECK-NEXT:    store <8 x i64> [[SHUFFLE3_I]], <8 x i64>* [[__T3_I]], align 64
// CHECK-NEXT:    [[TMP28:%.*]] = load <8 x i64>, <8 x i64>* [[__T2_I]], align 64
// CHECK-NEXT:    [[TMP29:%.*]] = load <8 x i64>, <8 x i64>* [[__T3_I]], align 64
// CHECK-NEXT:    store <8 x i64> [[TMP28]], <8 x i64>* [[__A_ADDR_I7_I]], align 64
// CHECK-NEXT:    store <8 x i64> [[TMP29]], <8 x i64>* [[__B_ADDR_I8_I]], align 64
// CHECK-NEXT:    [[TMP30:%.*]] = load <8 x i64>, <8 x i64>* [[__A_ADDR_I7_I]], align 64
// CHECK-NEXT:    [[TMP31:%.*]] = load <8 x i64>, <8 x i64>* [[__B_ADDR_I8_I]], align 64
// CHECK-NEXT:    [[TMP32:%.*]] = icmp sgt <8 x i64> [[TMP30]], [[TMP31]]
// CHECK-NEXT:    [[TMP33:%.*]] = select <8 x i1> [[TMP32]], <8 x i64> [[TMP30]], <8 x i64> [[TMP31]]
// CHECK-NEXT:    store <8 x i64> [[TMP33]], <8 x i64>* [[__T4_I]], align 64
// CHECK-NEXT:    [[TMP34:%.*]] = load <8 x i64>, <8 x i64>* [[__T4_I]], align 64
// CHECK-NEXT:    [[TMP35:%.*]] = load <8 x i64>, <8 x i64>* [[__T4_I]], align 64
// CHECK-NEXT:    [[SHUFFLE5_I:%.*]] = shufflevector <8 x i64> [[TMP34]], <8 x i64> [[TMP35]], <8 x i32> <i32 1, i32 0, i32 3, i32 2, i32 5, i32 4, i32 7, i32 6>
// CHECK-NEXT:    store <8 x i64> [[SHUFFLE5_I]], <8 x i64>* [[__T5_I]], align 64
// CHECK-NEXT:    [[TMP36:%.*]] = load <8 x i64>, <8 x i64>* [[__T4_I]], align 64
// CHECK-NEXT:    [[TMP37:%.*]] = load <8 x i64>, <8 x i64>* [[__T5_I]], align 64
// CHECK-NEXT:    store <8 x i64> [[TMP36]], <8 x i64>* [[__A_ADDR_I_I]], align 64
// CHECK-NEXT:    store <8 x i64> [[TMP37]], <8 x i64>* [[__B_ADDR_I_I]], align 64
// CHECK-NEXT:    [[TMP38:%.*]] = load <8 x i64>, <8 x i64>* [[__A_ADDR_I_I]], align 64
// CHECK-NEXT:    [[TMP39:%.*]] = load <8 x i64>, <8 x i64>* [[__B_ADDR_I_I]], align 64
// CHECK-NEXT:    [[TMP40:%.*]] = icmp sgt <8 x i64> [[TMP38]], [[TMP39]]
// CHECK-NEXT:    [[TMP41:%.*]] = select <8 x i1> [[TMP40]], <8 x i64> [[TMP38]], <8 x i64> [[TMP39]]
// CHECK-NEXT:    store <8 x i64> [[TMP41]], <8 x i64>* [[__T6_I]], align 64
// CHECK-NEXT:    [[TMP42:%.*]] = load <8 x i64>, <8 x i64>* [[__T6_I]], align 64
// CHECK-NEXT:    [[VECEXT_I:%.*]] = extractelement <8 x i64> [[TMP42]], i32 0
// CHECK-NEXT:    ret i64 [[VECEXT_I]]
long long test_mm512_mask_reduce_max_epi64(__mmask8 __M, __m512i __W){
  return _mm512_mask_reduce_max_epi64(__M, __W); 
}

// CHECK-LABEL: define i64 @test_mm512_mask_reduce_max_epu64(i8 zeroext %__M, <8 x i64> %__W) #0 {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[__A_ADDR_I9_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__B_ADDR_I10_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__A_ADDR_I7_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__B_ADDR_I8_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__A_ADDR_I6_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__B_ADDR_I_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[DOTCOMPOUNDLITERAL_I_I_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__U_ADDR_I_I:%.*]] = alloca i8, align 1
// CHECK-NEXT:    [[__A_ADDR_I_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__M_ADDR_I:%.*]] = alloca i8, align 1
// CHECK-NEXT:    [[__V_ADDR_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__T1_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__T2_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__T3_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__T4_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__T5_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__T6_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__M_ADDR:%.*]] = alloca i8, align 1
// CHECK-NEXT:    [[__W_ADDR:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    store i8 [[__M:%.*]], i8* [[__M_ADDR]], align 1
// CHECK-NEXT:    store <8 x i64> [[__W:%.*]], <8 x i64>* [[__W_ADDR]], align 64
// CHECK-NEXT:    [[TMP0:%.*]] = load i8, i8* [[__M_ADDR]], align 1
// CHECK-NEXT:    [[TMP1:%.*]] = load <8 x i64>, <8 x i64>* [[__W_ADDR]], align 64
// CHECK-NEXT:    store i8 [[TMP0]], i8* [[__M_ADDR_I]], align 1
// CHECK-NEXT:    store <8 x i64> [[TMP1]], <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK-NEXT:    [[TMP2:%.*]] = load i8, i8* [[__M_ADDR_I]], align 1
// CHECK-NEXT:    [[TMP3:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK-NEXT:    store i8 [[TMP2]], i8* [[__U_ADDR_I_I]], align 1
// CHECK-NEXT:    store <8 x i64> [[TMP3]], <8 x i64>* [[__A_ADDR_I_I]], align 64
// CHECK-NEXT:    [[TMP4:%.*]] = load i8, i8* [[__U_ADDR_I_I]], align 1
// CHECK-NEXT:    [[TMP5:%.*]] = load <8 x i64>, <8 x i64>* [[__A_ADDR_I_I]], align 64
// CHECK-NEXT:    store <8 x i64> zeroinitializer, <8 x i64>* [[DOTCOMPOUNDLITERAL_I_I_I]], align 64
// CHECK-NEXT:    [[TMP6:%.*]] = load <8 x i64>, <8 x i64>* [[DOTCOMPOUNDLITERAL_I_I_I]], align 64
// CHECK-NEXT:    [[TMP7:%.*]] = bitcast i8 [[TMP4]] to <8 x i1>
// CHECK-NEXT:    [[TMP8:%.*]] = select <8 x i1> [[TMP7]], <8 x i64> [[TMP5]], <8 x i64> [[TMP6]]
// CHECK-NEXT:    store <8 x i64> [[TMP8]], <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK-NEXT:    [[TMP9:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK-NEXT:    [[TMP10:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK-NEXT:    [[SHUFFLE_I:%.*]] = shufflevector <8 x i64> [[TMP9]], <8 x i64> [[TMP10]], <8 x i32> <i32 4, i32 5, i32 6, i32 7, i32 0, i32 1, i32 2, i32 3>
// CHECK-NEXT:    store <8 x i64> [[SHUFFLE_I]], <8 x i64>* [[__T1_I]], align 64
// CHECK-NEXT:    [[TMP11:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK-NEXT:    [[TMP12:%.*]] = load <8 x i64>, <8 x i64>* [[__T1_I]], align 64
// CHECK-NEXT:    store <8 x i64> [[TMP11]], <8 x i64>* [[__A_ADDR_I9_I]], align 64
// CHECK-NEXT:    store <8 x i64> [[TMP12]], <8 x i64>* [[__B_ADDR_I10_I]], align 64
// CHECK-NEXT:    [[TMP13:%.*]] = load <8 x i64>, <8 x i64>* [[__A_ADDR_I9_I]], align 64
// CHECK-NEXT:    [[TMP14:%.*]] = load <8 x i64>, <8 x i64>* [[__B_ADDR_I10_I]], align 64
// CHECK-NEXT:    [[TMP15:%.*]] = icmp ugt <8 x i64> [[TMP13]], [[TMP14]]
// CHECK-NEXT:    [[TMP16:%.*]] = select <8 x i1> [[TMP15]], <8 x i64> [[TMP13]], <8 x i64> [[TMP14]]
// CHECK-NEXT:    store <8 x i64> [[TMP16]], <8 x i64>* [[__T2_I]], align 64
// CHECK-NEXT:    [[TMP17:%.*]] = load <8 x i64>, <8 x i64>* [[__T2_I]], align 64
// CHECK-NEXT:    [[TMP18:%.*]] = load <8 x i64>, <8 x i64>* [[__T2_I]], align 64
// CHECK-NEXT:    [[SHUFFLE2_I:%.*]] = shufflevector <8 x i64> [[TMP17]], <8 x i64> [[TMP18]], <8 x i32> <i32 2, i32 3, i32 0, i32 1, i32 6, i32 7, i32 4, i32 5>
// CHECK-NEXT:    store <8 x i64> [[SHUFFLE2_I]], <8 x i64>* [[__T3_I]], align 64
// CHECK-NEXT:    [[TMP19:%.*]] = load <8 x i64>, <8 x i64>* [[__T2_I]], align 64
// CHECK-NEXT:    [[TMP20:%.*]] = load <8 x i64>, <8 x i64>* [[__T3_I]], align 64
// CHECK-NEXT:    store <8 x i64> [[TMP19]], <8 x i64>* [[__A_ADDR_I7_I]], align 64
// CHECK-NEXT:    store <8 x i64> [[TMP20]], <8 x i64>* [[__B_ADDR_I8_I]], align 64
// CHECK-NEXT:    [[TMP21:%.*]] = load <8 x i64>, <8 x i64>* [[__A_ADDR_I7_I]], align 64
// CHECK-NEXT:    [[TMP22:%.*]] = load <8 x i64>, <8 x i64>* [[__B_ADDR_I8_I]], align 64
// CHECK-NEXT:    [[TMP23:%.*]] = icmp ugt <8 x i64> [[TMP21]], [[TMP22]]
// CHECK-NEXT:    [[TMP24:%.*]] = select <8 x i1> [[TMP23]], <8 x i64> [[TMP21]], <8 x i64> [[TMP22]]
// CHECK-NEXT:    store <8 x i64> [[TMP24]], <8 x i64>* [[__T4_I]], align 64
// CHECK-NEXT:    [[TMP25:%.*]] = load <8 x i64>, <8 x i64>* [[__T4_I]], align 64
// CHECK-NEXT:    [[TMP26:%.*]] = load <8 x i64>, <8 x i64>* [[__T4_I]], align 64
// CHECK-NEXT:    [[SHUFFLE4_I:%.*]] = shufflevector <8 x i64> [[TMP25]], <8 x i64> [[TMP26]], <8 x i32> <i32 1, i32 0, i32 3, i32 2, i32 5, i32 4, i32 7, i32 6>
// CHECK-NEXT:    store <8 x i64> [[SHUFFLE4_I]], <8 x i64>* [[__T5_I]], align 64
// CHECK-NEXT:    [[TMP27:%.*]] = load <8 x i64>, <8 x i64>* [[__T4_I]], align 64
// CHECK-NEXT:    [[TMP28:%.*]] = load <8 x i64>, <8 x i64>* [[__T5_I]], align 64
// CHECK-NEXT:    store <8 x i64> [[TMP27]], <8 x i64>* [[__A_ADDR_I6_I]], align 64
// CHECK-NEXT:    store <8 x i64> [[TMP28]], <8 x i64>* [[__B_ADDR_I_I]], align 64
// CHECK-NEXT:    [[TMP29:%.*]] = load <8 x i64>, <8 x i64>* [[__A_ADDR_I6_I]], align 64
// CHECK-NEXT:    [[TMP30:%.*]] = load <8 x i64>, <8 x i64>* [[__B_ADDR_I_I]], align 64
// CHECK-NEXT:    [[TMP31:%.*]] = icmp ugt <8 x i64> [[TMP29]], [[TMP30]]
// CHECK-NEXT:    [[TMP32:%.*]] = select <8 x i1> [[TMP31]], <8 x i64> [[TMP29]], <8 x i64> [[TMP30]]
// CHECK-NEXT:    store <8 x i64> [[TMP32]], <8 x i64>* [[__T6_I]], align 64
// CHECK-NEXT:    [[TMP33:%.*]] = load <8 x i64>, <8 x i64>* [[__T6_I]], align 64
// CHECK-NEXT:    [[VECEXT_I:%.*]] = extractelement <8 x i64> [[TMP33]], i32 0
// CHECK-NEXT:    ret i64 [[VECEXT_I]]
unsigned long test_mm512_mask_reduce_max_epu64(__mmask8 __M, __m512i __W){
  return _mm512_mask_reduce_max_epu64(__M, __W); 
}

// CHECK-LABEL: define double @test_mm512_mask_reduce_max_pd(i8 zeroext %__M, <8 x double> %__W) #0 {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[__W2_ADDR_I_I:%.*]] = alloca <8 x double>, align 64
// CHECK-NEXT:    [[__U_ADDR_I_I:%.*]] = alloca i8, align 1
// CHECK-NEXT:    [[__A_ADDR_I_I:%.*]] = alloca <8 x double>, align 64
// CHECK-NEXT:    [[__A_ADDR_I12_I:%.*]] = alloca <4 x double>, align 32
// CHECK-NEXT:    [[__B_ADDR_I13_I:%.*]] = alloca <4 x double>, align 32
// CHECK-NEXT:    [[__A_ADDR_I10_I:%.*]] = alloca <2 x double>, align 16
// CHECK-NEXT:    [[__B_ADDR_I11_I:%.*]] = alloca <2 x double>, align 16
// CHECK-NEXT:    [[__A2_ADDR_I_I:%.*]] = alloca <2 x double>, align 16
// CHECK-NEXT:    [[__B_ADDR_I_I:%.*]] = alloca <2 x double>, align 16
// CHECK-NEXT:    [[__W_ADDR_I_I:%.*]] = alloca double, align 8
// CHECK-NEXT:    [[DOTCOMPOUNDLITERAL_I_I:%.*]] = alloca <8 x double>, align 64
// CHECK-NEXT:    [[__M_ADDR_I:%.*]] = alloca i8, align 1
// CHECK-NEXT:    [[__V_ADDR_I:%.*]] = alloca <8 x double>, align 64
// CHECK-NEXT:    [[__T1_I:%.*]] = alloca <4 x double>, align 32
// CHECK-NEXT:    [[__T2_I:%.*]] = alloca <4 x double>, align 32
// CHECK-NEXT:    [[__T3_I:%.*]] = alloca <4 x double>, align 32
// CHECK-NEXT:    [[__T4_I:%.*]] = alloca <2 x double>, align 16
// CHECK-NEXT:    [[__T5_I:%.*]] = alloca <2 x double>, align 16
// CHECK-NEXT:    [[__T6_I:%.*]] = alloca <2 x double>, align 16
// CHECK-NEXT:    [[__T7_I:%.*]] = alloca <2 x double>, align 16
// CHECK-NEXT:    [[__T8_I:%.*]] = alloca <2 x double>, align 16
// CHECK-NEXT:    [[__M_ADDR:%.*]] = alloca i8, align 1
// CHECK-NEXT:    [[__W_ADDR:%.*]] = alloca <8 x double>, align 64
// CHECK-NEXT:    store i8 [[__M:%.*]], i8* [[__M_ADDR]], align 1
// CHECK-NEXT:    store <8 x double> [[__W:%.*]], <8 x double>* [[__W_ADDR]], align 64
// CHECK-NEXT:    [[TMP0:%.*]] = load i8, i8* [[__M_ADDR]], align 1
// CHECK-NEXT:    [[TMP1:%.*]] = load <8 x double>, <8 x double>* [[__W_ADDR]], align 64
// CHECK-NEXT:    store i8 [[TMP0]], i8* [[__M_ADDR_I]], align 1
// CHECK-NEXT:    store <8 x double> [[TMP1]], <8 x double>* [[__V_ADDR_I]], align 64
// CHECK-NEXT:    store double 0xFFF0000000000000, double* [[__W_ADDR_I_I]], align 8
// CHECK-NEXT:    [[TMP2:%.*]] = load double, double* [[__W_ADDR_I_I]], align 8
// CHECK-NEXT:    [[VECINIT_I_I:%.*]] = insertelement <8 x double> undef, double [[TMP2]], i32 0
// CHECK-NEXT:    [[TMP3:%.*]] = load double, double* [[__W_ADDR_I_I]], align 8
// CHECK-NEXT:    [[VECINIT1_I_I:%.*]] = insertelement <8 x double> [[VECINIT_I_I]], double [[TMP3]], i32 1
// CHECK-NEXT:    [[TMP4:%.*]] = load double, double* [[__W_ADDR_I_I]], align 8
// CHECK-NEXT:    [[VECINIT2_I_I:%.*]] = insertelement <8 x double> [[VECINIT1_I_I]], double [[TMP4]], i32 2
// CHECK-NEXT:    [[TMP5:%.*]] = load double, double* [[__W_ADDR_I_I]], align 8
// CHECK-NEXT:    [[VECINIT3_I_I:%.*]] = insertelement <8 x double> [[VECINIT2_I_I]], double [[TMP5]], i32 3
// CHECK-NEXT:    [[TMP6:%.*]] = load double, double* [[__W_ADDR_I_I]], align 8
// CHECK-NEXT:    [[VECINIT4_I_I:%.*]] = insertelement <8 x double> [[VECINIT3_I_I]], double [[TMP6]], i32 4
// CHECK-NEXT:    [[TMP7:%.*]] = load double, double* [[__W_ADDR_I_I]], align 8
// CHECK-NEXT:    [[VECINIT5_I_I:%.*]] = insertelement <8 x double> [[VECINIT4_I_I]], double [[TMP7]], i32 5
// CHECK-NEXT:    [[TMP8:%.*]] = load double, double* [[__W_ADDR_I_I]], align 8
// CHECK-NEXT:    [[VECINIT6_I_I:%.*]] = insertelement <8 x double> [[VECINIT5_I_I]], double [[TMP8]], i32 6
// CHECK-NEXT:    [[TMP9:%.*]] = load double, double* [[__W_ADDR_I_I]], align 8
// CHECK-NEXT:    [[VECINIT7_I_I:%.*]] = insertelement <8 x double> [[VECINIT6_I_I]], double [[TMP9]], i32 7
// CHECK-NEXT:    store <8 x double> [[VECINIT7_I_I]], <8 x double>* [[DOTCOMPOUNDLITERAL_I_I]], align 64
// CHECK-NEXT:    [[TMP10:%.*]] = load <8 x double>, <8 x double>* [[DOTCOMPOUNDLITERAL_I_I]], align 64
// CHECK-NEXT:    [[TMP11:%.*]] = load i8, i8* [[__M_ADDR_I]], align 1
// CHECK-NEXT:    [[TMP12:%.*]] = load <8 x double>, <8 x double>* [[__V_ADDR_I]], align 64
// CHECK-NEXT:    store <8 x double> [[TMP10]], <8 x double>* [[__W2_ADDR_I_I]], align 64
// CHECK-NEXT:    store i8 [[TMP11]], i8* [[__U_ADDR_I_I]], align 1
// CHECK-NEXT:    store <8 x double> [[TMP12]], <8 x double>* [[__A_ADDR_I_I]], align 64
// CHECK-NEXT:    [[TMP13:%.*]] = load i8, i8* [[__U_ADDR_I_I]], align 1
// CHECK-NEXT:    [[TMP14:%.*]] = load <8 x double>, <8 x double>* [[__A_ADDR_I_I]], align 64
// CHECK-NEXT:    [[TMP15:%.*]] = load <8 x double>, <8 x double>* [[__W2_ADDR_I_I]], align 64
// CHECK-NEXT:    [[TMP16:%.*]] = bitcast i8 [[TMP13]] to <8 x i1>
// CHECK-NEXT:    [[TMP17:%.*]] = select <8 x i1> [[TMP16]], <8 x double> [[TMP14]], <8 x double> [[TMP15]]
// CHECK-NEXT:    store <8 x double> [[TMP17]], <8 x double>* [[__V_ADDR_I]], align 64
// CHECK-NEXT:    [[TMP18:%.*]] = load <8 x double>, <8 x double>* [[__V_ADDR_I]], align 64
// CHECK-NEXT:    [[EXTRACT_I:%.*]] = shufflevector <8 x double> [[TMP18]], <8 x double> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// CHECK-NEXT:    store <4 x double> [[EXTRACT_I]], <4 x double>* [[__T1_I]], align 32
// CHECK-NEXT:    [[TMP19:%.*]] = load <8 x double>, <8 x double>* [[__V_ADDR_I]], align 64
// CHECK-NEXT:    [[EXTRACT4_I:%.*]] = shufflevector <8 x double> [[TMP19]], <8 x double> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK-NEXT:    store <4 x double> [[EXTRACT4_I]], <4 x double>* [[__T2_I]], align 32
// CHECK-NEXT:    [[TMP20:%.*]] = load <4 x double>, <4 x double>* [[__T1_I]], align 32
// CHECK-NEXT:    [[TMP21:%.*]] = load <4 x double>, <4 x double>* [[__T2_I]], align 32
// CHECK-NEXT:    store <4 x double> [[TMP20]], <4 x double>* [[__A_ADDR_I12_I]], align 32
// CHECK-NEXT:    store <4 x double> [[TMP21]], <4 x double>* [[__B_ADDR_I13_I]], align 32
// CHECK-NEXT:    [[TMP22:%.*]] = load <4 x double>, <4 x double>* [[__A_ADDR_I12_I]], align 32
// CHECK-NEXT:    [[TMP23:%.*]] = load <4 x double>, <4 x double>* [[__B_ADDR_I13_I]], align 32
// CHECK-NEXT:    [[TMP24:%.*]] = call <4 x double> @llvm.x86.avx.max.pd.256(<4 x double> [[TMP22]], <4 x double> [[TMP23]]) #2
// CHECK-NEXT:    store <4 x double> [[TMP24]], <4 x double>* [[__T3_I]], align 32
// CHECK-NEXT:    [[TMP25:%.*]] = load <4 x double>, <4 x double>* [[__T3_I]], align 32
// CHECK-NEXT:    [[EXTRACT6_I:%.*]] = shufflevector <4 x double> [[TMP25]], <4 x double> undef, <2 x i32> <i32 0, i32 1>
// CHECK-NEXT:    store <2 x double> [[EXTRACT6_I]], <2 x double>* [[__T4_I]], align 16
// CHECK-NEXT:    [[TMP26:%.*]] = load <4 x double>, <4 x double>* [[__T3_I]], align 32
// CHECK-NEXT:    [[EXTRACT7_I:%.*]] = shufflevector <4 x double> [[TMP26]], <4 x double> undef, <2 x i32> <i32 2, i32 3>
// CHECK-NEXT:    store <2 x double> [[EXTRACT7_I]], <2 x double>* [[__T5_I]], align 16
// CHECK-NEXT:    [[TMP27:%.*]] = load <2 x double>, <2 x double>* [[__T4_I]], align 16
// CHECK-NEXT:    [[TMP28:%.*]] = load <2 x double>, <2 x double>* [[__T5_I]], align 16
// CHECK-NEXT:    store <2 x double> [[TMP27]], <2 x double>* [[__A_ADDR_I10_I]], align 16
// CHECK-NEXT:    store <2 x double> [[TMP28]], <2 x double>* [[__B_ADDR_I11_I]], align 16
// CHECK-NEXT:    [[TMP29:%.*]] = load <2 x double>, <2 x double>* [[__A_ADDR_I10_I]], align 16
// CHECK-NEXT:    [[TMP30:%.*]] = load <2 x double>, <2 x double>* [[__B_ADDR_I11_I]], align 16
// CHECK-NEXT:    [[TMP31:%.*]] = call <2 x double> @llvm.x86.sse2.max.pd(<2 x double> [[TMP29]], <2 x double> [[TMP30]]) #2
// CHECK-NEXT:    store <2 x double> [[TMP31]], <2 x double>* [[__T6_I]], align 16
// CHECK-NEXT:    [[TMP32:%.*]] = load <2 x double>, <2 x double>* [[__T6_I]], align 16
// CHECK-NEXT:    [[TMP33:%.*]] = load <2 x double>, <2 x double>* [[__T6_I]], align 16
// CHECK-NEXT:    [[SHUFFLE_I:%.*]] = shufflevector <2 x double> [[TMP32]], <2 x double> [[TMP33]], <2 x i32> <i32 1, i32 0>
// CHECK-NEXT:    store <2 x double> [[SHUFFLE_I]], <2 x double>* [[__T7_I]], align 16
// CHECK-NEXT:    [[TMP34:%.*]] = load <2 x double>, <2 x double>* [[__T6_I]], align 16
// CHECK-NEXT:    [[TMP35:%.*]] = load <2 x double>, <2 x double>* [[__T7_I]], align 16
// CHECK-NEXT:    store <2 x double> [[TMP34]], <2 x double>* [[__A2_ADDR_I_I]], align 16
// CHECK-NEXT:    store <2 x double> [[TMP35]], <2 x double>* [[__B_ADDR_I_I]], align 16
// CHECK-NEXT:    [[TMP36:%.*]] = load <2 x double>, <2 x double>* [[__A2_ADDR_I_I]], align 16
// CHECK-NEXT:    [[TMP37:%.*]] = load <2 x double>, <2 x double>* [[__B_ADDR_I_I]], align 16
// CHECK-NEXT:    [[TMP38:%.*]] = call <2 x double> @llvm.x86.sse2.max.pd(<2 x double> [[TMP36]], <2 x double> [[TMP37]]) #2
// CHECK-NEXT:    store <2 x double> [[TMP38]], <2 x double>* [[__T8_I]], align 16
// CHECK-NEXT:    [[TMP39:%.*]] = load <2 x double>, <2 x double>* [[__T8_I]], align 16
// CHECK-NEXT:    [[VECEXT_I:%.*]] = extractelement <2 x double> [[TMP39]], i32 0
// CHECK-NEXT:    ret double [[VECEXT_I]]
double test_mm512_mask_reduce_max_pd(__mmask8 __M, __m512d __W){
  return _mm512_mask_reduce_max_pd(__M, __W); 
}

// CHECK-LABEL: define i64 @test_mm512_mask_reduce_min_epi64(i8 zeroext %__M, <8 x i64> %__W) #0 {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[__W_ADDR_I_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__U_ADDR_I_I:%.*]] = alloca i8, align 1
// CHECK-NEXT:    [[__A_ADDR_I11_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__A_ADDR_I9_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__B_ADDR_I10_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__A_ADDR_I7_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__B_ADDR_I8_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__A_ADDR_I_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__B_ADDR_I_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__D_ADDR_I_I:%.*]] = alloca i64, align 8
// CHECK-NEXT:    [[DOTCOMPOUNDLITERAL_I_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__M_ADDR_I:%.*]] = alloca i8, align 1
// CHECK-NEXT:    [[__V_ADDR_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__T1_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__T2_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__T3_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__T4_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__T5_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__T6_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__M_ADDR:%.*]] = alloca i8, align 1
// CHECK-NEXT:    [[__W_ADDR:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    store i8 [[__M:%.*]], i8* [[__M_ADDR]], align 1
// CHECK-NEXT:    store <8 x i64> [[__W:%.*]], <8 x i64>* [[__W_ADDR]], align 64
// CHECK-NEXT:    [[TMP0:%.*]] = load i8, i8* [[__M_ADDR]], align 1
// CHECK-NEXT:    [[TMP1:%.*]] = load <8 x i64>, <8 x i64>* [[__W_ADDR]], align 64
// CHECK-NEXT:    store i8 [[TMP0]], i8* [[__M_ADDR_I]], align 1
// CHECK-NEXT:    store <8 x i64> [[TMP1]], <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK-NEXT:    store i64 9223372036854775807, i64* [[__D_ADDR_I_I]], align 8
// CHECK-NEXT:    [[TMP2:%.*]] = load i64, i64* [[__D_ADDR_I_I]], align 8
// CHECK-NEXT:    [[VECINIT_I_I:%.*]] = insertelement <8 x i64> undef, i64 [[TMP2]], i32 0
// CHECK-NEXT:    [[TMP3:%.*]] = load i64, i64* [[__D_ADDR_I_I]], align 8
// CHECK-NEXT:    [[VECINIT1_I_I:%.*]] = insertelement <8 x i64> [[VECINIT_I_I]], i64 [[TMP3]], i32 1
// CHECK-NEXT:    [[TMP4:%.*]] = load i64, i64* [[__D_ADDR_I_I]], align 8
// CHECK-NEXT:    [[VECINIT2_I_I:%.*]] = insertelement <8 x i64> [[VECINIT1_I_I]], i64 [[TMP4]], i32 2
// CHECK-NEXT:    [[TMP5:%.*]] = load i64, i64* [[__D_ADDR_I_I]], align 8
// CHECK-NEXT:    [[VECINIT3_I_I:%.*]] = insertelement <8 x i64> [[VECINIT2_I_I]], i64 [[TMP5]], i32 3
// CHECK-NEXT:    [[TMP6:%.*]] = load i64, i64* [[__D_ADDR_I_I]], align 8
// CHECK-NEXT:    [[VECINIT4_I_I:%.*]] = insertelement <8 x i64> [[VECINIT3_I_I]], i64 [[TMP6]], i32 4
// CHECK-NEXT:    [[TMP7:%.*]] = load i64, i64* [[__D_ADDR_I_I]], align 8
// CHECK-NEXT:    [[VECINIT5_I_I:%.*]] = insertelement <8 x i64> [[VECINIT4_I_I]], i64 [[TMP7]], i32 5
// CHECK-NEXT:    [[TMP8:%.*]] = load i64, i64* [[__D_ADDR_I_I]], align 8
// CHECK-NEXT:    [[VECINIT6_I_I:%.*]] = insertelement <8 x i64> [[VECINIT5_I_I]], i64 [[TMP8]], i32 6
// CHECK-NEXT:    [[TMP9:%.*]] = load i64, i64* [[__D_ADDR_I_I]], align 8
// CHECK-NEXT:    [[VECINIT7_I_I:%.*]] = insertelement <8 x i64> [[VECINIT6_I_I]], i64 [[TMP9]], i32 7
// CHECK-NEXT:    store <8 x i64> [[VECINIT7_I_I]], <8 x i64>* [[DOTCOMPOUNDLITERAL_I_I]], align 64
// CHECK-NEXT:    [[TMP10:%.*]] = load <8 x i64>, <8 x i64>* [[DOTCOMPOUNDLITERAL_I_I]], align 64
// CHECK-NEXT:    [[TMP11:%.*]] = load i8, i8* [[__M_ADDR_I]], align 1
// CHECK-NEXT:    [[TMP12:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK-NEXT:    store <8 x i64> [[TMP10]], <8 x i64>* [[__W_ADDR_I_I]], align 64
// CHECK-NEXT:    store i8 [[TMP11]], i8* [[__U_ADDR_I_I]], align 1
// CHECK-NEXT:    store <8 x i64> [[TMP12]], <8 x i64>* [[__A_ADDR_I11_I]], align 64
// CHECK-NEXT:    [[TMP13:%.*]] = load i8, i8* [[__U_ADDR_I_I]], align 1
// CHECK-NEXT:    [[TMP14:%.*]] = load <8 x i64>, <8 x i64>* [[__A_ADDR_I11_I]], align 64
// CHECK-NEXT:    [[TMP15:%.*]] = load <8 x i64>, <8 x i64>* [[__W_ADDR_I_I]], align 64
// CHECK-NEXT:    [[TMP16:%.*]] = bitcast i8 [[TMP13]] to <8 x i1>
// CHECK-NEXT:    [[TMP17:%.*]] = select <8 x i1> [[TMP16]], <8 x i64> [[TMP14]], <8 x i64> [[TMP15]]
// CHECK-NEXT:    store <8 x i64> [[TMP17]], <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK-NEXT:    [[TMP18:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK-NEXT:    [[TMP19:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK-NEXT:    [[SHUFFLE_I:%.*]] = shufflevector <8 x i64> [[TMP18]], <8 x i64> [[TMP19]], <8 x i32> <i32 4, i32 5, i32 6, i32 7, i32 0, i32 1, i32 2, i32 3>
// CHECK-NEXT:    store <8 x i64> [[SHUFFLE_I]], <8 x i64>* [[__T1_I]], align 64
// CHECK-NEXT:    [[TMP20:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK-NEXT:    [[TMP21:%.*]] = load <8 x i64>, <8 x i64>* [[__T1_I]], align 64
// CHECK-NEXT:    store <8 x i64> [[TMP20]], <8 x i64>* [[__A_ADDR_I9_I]], align 64
// CHECK-NEXT:    store <8 x i64> [[TMP21]], <8 x i64>* [[__B_ADDR_I10_I]], align 64
// CHECK-NEXT:    [[TMP22:%.*]] = load <8 x i64>, <8 x i64>* [[__A_ADDR_I9_I]], align 64
// CHECK-NEXT:    [[TMP23:%.*]] = load <8 x i64>, <8 x i64>* [[__B_ADDR_I10_I]], align 64
// CHECK-NEXT:    [[TMP24:%.*]] = icmp slt <8 x i64> [[TMP22]], [[TMP23]]
// CHECK-NEXT:    [[TMP25:%.*]] = select <8 x i1> [[TMP24]], <8 x i64> [[TMP22]], <8 x i64> [[TMP23]]
// CHECK-NEXT:    store <8 x i64> [[TMP25]], <8 x i64>* [[__T2_I]], align 64
// CHECK-NEXT:    [[TMP26:%.*]] = load <8 x i64>, <8 x i64>* [[__T2_I]], align 64
// CHECK-NEXT:    [[TMP27:%.*]] = load <8 x i64>, <8 x i64>* [[__T2_I]], align 64
// CHECK-NEXT:    [[SHUFFLE3_I:%.*]] = shufflevector <8 x i64> [[TMP26]], <8 x i64> [[TMP27]], <8 x i32> <i32 2, i32 3, i32 0, i32 1, i32 6, i32 7, i32 4, i32 5>
// CHECK-NEXT:    store <8 x i64> [[SHUFFLE3_I]], <8 x i64>* [[__T3_I]], align 64
// CHECK-NEXT:    [[TMP28:%.*]] = load <8 x i64>, <8 x i64>* [[__T2_I]], align 64
// CHECK-NEXT:    [[TMP29:%.*]] = load <8 x i64>, <8 x i64>* [[__T3_I]], align 64
// CHECK-NEXT:    store <8 x i64> [[TMP28]], <8 x i64>* [[__A_ADDR_I7_I]], align 64
// CHECK-NEXT:    store <8 x i64> [[TMP29]], <8 x i64>* [[__B_ADDR_I8_I]], align 64
// CHECK-NEXT:    [[TMP30:%.*]] = load <8 x i64>, <8 x i64>* [[__A_ADDR_I7_I]], align 64
// CHECK-NEXT:    [[TMP31:%.*]] = load <8 x i64>, <8 x i64>* [[__B_ADDR_I8_I]], align 64
// CHECK-NEXT:    [[TMP32:%.*]] = icmp slt <8 x i64> [[TMP30]], [[TMP31]]
// CHECK-NEXT:    [[TMP33:%.*]] = select <8 x i1> [[TMP32]], <8 x i64> [[TMP30]], <8 x i64> [[TMP31]]
// CHECK-NEXT:    store <8 x i64> [[TMP33]], <8 x i64>* [[__T4_I]], align 64
// CHECK-NEXT:    [[TMP34:%.*]] = load <8 x i64>, <8 x i64>* [[__T4_I]], align 64
// CHECK-NEXT:    [[TMP35:%.*]] = load <8 x i64>, <8 x i64>* [[__T4_I]], align 64
// CHECK-NEXT:    [[SHUFFLE5_I:%.*]] = shufflevector <8 x i64> [[TMP34]], <8 x i64> [[TMP35]], <8 x i32> <i32 1, i32 0, i32 3, i32 2, i32 5, i32 4, i32 7, i32 6>
// CHECK-NEXT:    store <8 x i64> [[SHUFFLE5_I]], <8 x i64>* [[__T5_I]], align 64
// CHECK-NEXT:    [[TMP36:%.*]] = load <8 x i64>, <8 x i64>* [[__T4_I]], align 64
// CHECK-NEXT:    [[TMP37:%.*]] = load <8 x i64>, <8 x i64>* [[__T5_I]], align 64
// CHECK-NEXT:    store <8 x i64> [[TMP36]], <8 x i64>* [[__A_ADDR_I_I]], align 64
// CHECK-NEXT:    store <8 x i64> [[TMP37]], <8 x i64>* [[__B_ADDR_I_I]], align 64
// CHECK-NEXT:    [[TMP38:%.*]] = load <8 x i64>, <8 x i64>* [[__A_ADDR_I_I]], align 64
// CHECK-NEXT:    [[TMP39:%.*]] = load <8 x i64>, <8 x i64>* [[__B_ADDR_I_I]], align 64
// CHECK-NEXT:    [[TMP40:%.*]] = icmp slt <8 x i64> [[TMP38]], [[TMP39]]
// CHECK-NEXT:    [[TMP41:%.*]] = select <8 x i1> [[TMP40]], <8 x i64> [[TMP38]], <8 x i64> [[TMP39]]
// CHECK-NEXT:    store <8 x i64> [[TMP41]], <8 x i64>* [[__T6_I]], align 64
// CHECK-NEXT:    [[TMP42:%.*]] = load <8 x i64>, <8 x i64>* [[__T6_I]], align 64
// CHECK-NEXT:    [[VECEXT_I:%.*]] = extractelement <8 x i64> [[TMP42]], i32 0
// CHECK-NEXT:    ret i64 [[VECEXT_I]]
long long test_mm512_mask_reduce_min_epi64(__mmask8 __M, __m512i __W){
  return _mm512_mask_reduce_min_epi64(__M, __W); 
}

// CHECK-LABEL: define i64 @test_mm512_mask_reduce_min_epu64(i8 zeroext %__M, <8 x i64> %__W) #0 {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[__W_ADDR_I_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__U_ADDR_I_I:%.*]] = alloca i8, align 1
// CHECK-NEXT:    [[__A_ADDR_I11_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__A_ADDR_I9_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__B_ADDR_I10_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__A_ADDR_I7_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__B_ADDR_I8_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__A_ADDR_I_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__B_ADDR_I_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__D_ADDR_I_I:%.*]] = alloca i64, align 8
// CHECK-NEXT:    [[DOTCOMPOUNDLITERAL_I_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__M_ADDR_I:%.*]] = alloca i8, align 1
// CHECK-NEXT:    [[__V_ADDR_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__T1_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__T2_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__T3_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__T4_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__T5_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__T6_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__M_ADDR:%.*]] = alloca i8, align 1
// CHECK-NEXT:    [[__W_ADDR:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    store i8 [[__M:%.*]], i8* [[__M_ADDR]], align 1
// CHECK-NEXT:    store <8 x i64> [[__W:%.*]], <8 x i64>* [[__W_ADDR]], align 64
// CHECK-NEXT:    [[TMP0:%.*]] = load i8, i8* [[__M_ADDR]], align 1
// CHECK-NEXT:    [[TMP1:%.*]] = load <8 x i64>, <8 x i64>* [[__W_ADDR]], align 64
// CHECK-NEXT:    store i8 [[TMP0]], i8* [[__M_ADDR_I]], align 1
// CHECK-NEXT:    store <8 x i64> [[TMP1]], <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK-NEXT:    store i64 -1, i64* [[__D_ADDR_I_I]], align 8
// CHECK-NEXT:    [[TMP2:%.*]] = load i64, i64* [[__D_ADDR_I_I]], align 8
// CHECK-NEXT:    [[VECINIT_I_I:%.*]] = insertelement <8 x i64> undef, i64 [[TMP2]], i32 0
// CHECK-NEXT:    [[TMP3:%.*]] = load i64, i64* [[__D_ADDR_I_I]], align 8
// CHECK-NEXT:    [[VECINIT1_I_I:%.*]] = insertelement <8 x i64> [[VECINIT_I_I]], i64 [[TMP3]], i32 1
// CHECK-NEXT:    [[TMP4:%.*]] = load i64, i64* [[__D_ADDR_I_I]], align 8
// CHECK-NEXT:    [[VECINIT2_I_I:%.*]] = insertelement <8 x i64> [[VECINIT1_I_I]], i64 [[TMP4]], i32 2
// CHECK-NEXT:    [[TMP5:%.*]] = load i64, i64* [[__D_ADDR_I_I]], align 8
// CHECK-NEXT:    [[VECINIT3_I_I:%.*]] = insertelement <8 x i64> [[VECINIT2_I_I]], i64 [[TMP5]], i32 3
// CHECK-NEXT:    [[TMP6:%.*]] = load i64, i64* [[__D_ADDR_I_I]], align 8
// CHECK-NEXT:    [[VECINIT4_I_I:%.*]] = insertelement <8 x i64> [[VECINIT3_I_I]], i64 [[TMP6]], i32 4
// CHECK-NEXT:    [[TMP7:%.*]] = load i64, i64* [[__D_ADDR_I_I]], align 8
// CHECK-NEXT:    [[VECINIT5_I_I:%.*]] = insertelement <8 x i64> [[VECINIT4_I_I]], i64 [[TMP7]], i32 5
// CHECK-NEXT:    [[TMP8:%.*]] = load i64, i64* [[__D_ADDR_I_I]], align 8
// CHECK-NEXT:    [[VECINIT6_I_I:%.*]] = insertelement <8 x i64> [[VECINIT5_I_I]], i64 [[TMP8]], i32 6
// CHECK-NEXT:    [[TMP9:%.*]] = load i64, i64* [[__D_ADDR_I_I]], align 8
// CHECK-NEXT:    [[VECINIT7_I_I:%.*]] = insertelement <8 x i64> [[VECINIT6_I_I]], i64 [[TMP9]], i32 7
// CHECK-NEXT:    store <8 x i64> [[VECINIT7_I_I]], <8 x i64>* [[DOTCOMPOUNDLITERAL_I_I]], align 64
// CHECK-NEXT:    [[TMP10:%.*]] = load <8 x i64>, <8 x i64>* [[DOTCOMPOUNDLITERAL_I_I]], align 64
// CHECK-NEXT:    [[TMP11:%.*]] = load i8, i8* [[__M_ADDR_I]], align 1
// CHECK-NEXT:    [[TMP12:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK-NEXT:    store <8 x i64> [[TMP10]], <8 x i64>* [[__W_ADDR_I_I]], align 64
// CHECK-NEXT:    store i8 [[TMP11]], i8* [[__U_ADDR_I_I]], align 1
// CHECK-NEXT:    store <8 x i64> [[TMP12]], <8 x i64>* [[__A_ADDR_I11_I]], align 64
// CHECK-NEXT:    [[TMP13:%.*]] = load i8, i8* [[__U_ADDR_I_I]], align 1
// CHECK-NEXT:    [[TMP14:%.*]] = load <8 x i64>, <8 x i64>* [[__A_ADDR_I11_I]], align 64
// CHECK-NEXT:    [[TMP15:%.*]] = load <8 x i64>, <8 x i64>* [[__W_ADDR_I_I]], align 64
// CHECK-NEXT:    [[TMP16:%.*]] = bitcast i8 [[TMP13]] to <8 x i1>
// CHECK-NEXT:    [[TMP17:%.*]] = select <8 x i1> [[TMP16]], <8 x i64> [[TMP14]], <8 x i64> [[TMP15]]
// CHECK-NEXT:    store <8 x i64> [[TMP17]], <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK-NEXT:    [[TMP18:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK-NEXT:    [[TMP19:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK-NEXT:    [[SHUFFLE_I:%.*]] = shufflevector <8 x i64> [[TMP18]], <8 x i64> [[TMP19]], <8 x i32> <i32 4, i32 5, i32 6, i32 7, i32 0, i32 1, i32 2, i32 3>
// CHECK-NEXT:    store <8 x i64> [[SHUFFLE_I]], <8 x i64>* [[__T1_I]], align 64
// CHECK-NEXT:    [[TMP20:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK-NEXT:    [[TMP21:%.*]] = load <8 x i64>, <8 x i64>* [[__T1_I]], align 64
// CHECK-NEXT:    store <8 x i64> [[TMP20]], <8 x i64>* [[__A_ADDR_I9_I]], align 64
// CHECK-NEXT:    store <8 x i64> [[TMP21]], <8 x i64>* [[__B_ADDR_I10_I]], align 64
// CHECK-NEXT:    [[TMP22:%.*]] = load <8 x i64>, <8 x i64>* [[__A_ADDR_I9_I]], align 64
// CHECK-NEXT:    [[TMP23:%.*]] = load <8 x i64>, <8 x i64>* [[__B_ADDR_I10_I]], align 64
// CHECK-NEXT:    [[TMP24:%.*]] = icmp ult <8 x i64> [[TMP22]], [[TMP23]]
// CHECK-NEXT:    [[TMP25:%.*]] = select <8 x i1> [[TMP24]], <8 x i64> [[TMP22]], <8 x i64> [[TMP23]]
// CHECK-NEXT:    store <8 x i64> [[TMP25]], <8 x i64>* [[__T2_I]], align 64
// CHECK-NEXT:    [[TMP26:%.*]] = load <8 x i64>, <8 x i64>* [[__T2_I]], align 64
// CHECK-NEXT:    [[TMP27:%.*]] = load <8 x i64>, <8 x i64>* [[__T2_I]], align 64
// CHECK-NEXT:    [[SHUFFLE3_I:%.*]] = shufflevector <8 x i64> [[TMP26]], <8 x i64> [[TMP27]], <8 x i32> <i32 2, i32 3, i32 0, i32 1, i32 6, i32 7, i32 4, i32 5>
// CHECK-NEXT:    store <8 x i64> [[SHUFFLE3_I]], <8 x i64>* [[__T3_I]], align 64
// CHECK-NEXT:    [[TMP28:%.*]] = load <8 x i64>, <8 x i64>* [[__T2_I]], align 64
// CHECK-NEXT:    [[TMP29:%.*]] = load <8 x i64>, <8 x i64>* [[__T3_I]], align 64
// CHECK-NEXT:    store <8 x i64> [[TMP28]], <8 x i64>* [[__A_ADDR_I7_I]], align 64
// CHECK-NEXT:    store <8 x i64> [[TMP29]], <8 x i64>* [[__B_ADDR_I8_I]], align 64
// CHECK-NEXT:    [[TMP30:%.*]] = load <8 x i64>, <8 x i64>* [[__A_ADDR_I7_I]], align 64
// CHECK-NEXT:    [[TMP31:%.*]] = load <8 x i64>, <8 x i64>* [[__B_ADDR_I8_I]], align 64
// CHECK-NEXT:    [[TMP32:%.*]] = icmp ult <8 x i64> [[TMP30]], [[TMP31]]
// CHECK-NEXT:    [[TMP33:%.*]] = select <8 x i1> [[TMP32]], <8 x i64> [[TMP30]], <8 x i64> [[TMP31]]
// CHECK-NEXT:    store <8 x i64> [[TMP33]], <8 x i64>* [[__T4_I]], align 64
// CHECK-NEXT:    [[TMP34:%.*]] = load <8 x i64>, <8 x i64>* [[__T4_I]], align 64
// CHECK-NEXT:    [[TMP35:%.*]] = load <8 x i64>, <8 x i64>* [[__T4_I]], align 64
// CHECK-NEXT:    [[SHUFFLE5_I:%.*]] = shufflevector <8 x i64> [[TMP34]], <8 x i64> [[TMP35]], <8 x i32> <i32 1, i32 0, i32 3, i32 2, i32 5, i32 4, i32 7, i32 6>
// CHECK-NEXT:    store <8 x i64> [[SHUFFLE5_I]], <8 x i64>* [[__T5_I]], align 64
// CHECK-NEXT:    [[TMP36:%.*]] = load <8 x i64>, <8 x i64>* [[__T4_I]], align 64
// CHECK-NEXT:    [[TMP37:%.*]] = load <8 x i64>, <8 x i64>* [[__T5_I]], align 64
// CHECK-NEXT:    store <8 x i64> [[TMP36]], <8 x i64>* [[__A_ADDR_I_I]], align 64
// CHECK-NEXT:    store <8 x i64> [[TMP37]], <8 x i64>* [[__B_ADDR_I_I]], align 64
// CHECK-NEXT:    [[TMP38:%.*]] = load <8 x i64>, <8 x i64>* [[__A_ADDR_I_I]], align 64
// CHECK-NEXT:    [[TMP39:%.*]] = load <8 x i64>, <8 x i64>* [[__B_ADDR_I_I]], align 64
// CHECK-NEXT:    [[TMP40:%.*]] = icmp ult <8 x i64> [[TMP38]], [[TMP39]]
// CHECK-NEXT:    [[TMP41:%.*]] = select <8 x i1> [[TMP40]], <8 x i64> [[TMP38]], <8 x i64> [[TMP39]]
// CHECK-NEXT:    store <8 x i64> [[TMP41]], <8 x i64>* [[__T6_I]], align 64
// CHECK-NEXT:    [[TMP42:%.*]] = load <8 x i64>, <8 x i64>* [[__T6_I]], align 64
// CHECK-NEXT:    [[VECEXT_I:%.*]] = extractelement <8 x i64> [[TMP42]], i32 0
// CHECK-NEXT:    ret i64 [[VECEXT_I]]
long long test_mm512_mask_reduce_min_epu64(__mmask8 __M, __m512i __W){
  return _mm512_mask_reduce_min_epu64(__M, __W);
}

// CHECK-LABEL: define double @test_mm512_mask_reduce_min_pd(i8 zeroext %__M, <8 x double> %__W) #0 {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[__W2_ADDR_I_I:%.*]] = alloca <8 x double>, align 64
// CHECK-NEXT:    [[__U_ADDR_I_I:%.*]] = alloca i8, align 1
// CHECK-NEXT:    [[__A_ADDR_I_I:%.*]] = alloca <8 x double>, align 64
// CHECK-NEXT:    [[__A_ADDR_I12_I:%.*]] = alloca <4 x double>, align 32
// CHECK-NEXT:    [[__B_ADDR_I13_I:%.*]] = alloca <4 x double>, align 32
// CHECK-NEXT:    [[__A_ADDR_I10_I:%.*]] = alloca <2 x double>, align 16
// CHECK-NEXT:    [[__B_ADDR_I11_I:%.*]] = alloca <2 x double>, align 16
// CHECK-NEXT:    [[__A2_ADDR_I_I:%.*]] = alloca <2 x double>, align 16
// CHECK-NEXT:    [[__B_ADDR_I_I:%.*]] = alloca <2 x double>, align 16
// CHECK-NEXT:    [[__W_ADDR_I_I:%.*]] = alloca double, align 8
// CHECK-NEXT:    [[DOTCOMPOUNDLITERAL_I_I:%.*]] = alloca <8 x double>, align 64
// CHECK-NEXT:    [[__M_ADDR_I:%.*]] = alloca i8, align 1
// CHECK-NEXT:    [[__V_ADDR_I:%.*]] = alloca <8 x double>, align 64
// CHECK-NEXT:    [[__T1_I:%.*]] = alloca <4 x double>, align 32
// CHECK-NEXT:    [[__T2_I:%.*]] = alloca <4 x double>, align 32
// CHECK-NEXT:    [[__T3_I:%.*]] = alloca <4 x double>, align 32
// CHECK-NEXT:    [[__T4_I:%.*]] = alloca <2 x double>, align 16
// CHECK-NEXT:    [[__T5_I:%.*]] = alloca <2 x double>, align 16
// CHECK-NEXT:    [[__T6_I:%.*]] = alloca <2 x double>, align 16
// CHECK-NEXT:    [[__T7_I:%.*]] = alloca <2 x double>, align 16
// CHECK-NEXT:    [[__T8_I:%.*]] = alloca <2 x double>, align 16
// CHECK-NEXT:    [[__M_ADDR:%.*]] = alloca i8, align 1
// CHECK-NEXT:    [[__W_ADDR:%.*]] = alloca <8 x double>, align 64
// CHECK-NEXT:    store i8 [[__M:%.*]], i8* [[__M_ADDR]], align 1
// CHECK-NEXT:    store <8 x double> [[__W:%.*]], <8 x double>* [[__W_ADDR]], align 64
// CHECK-NEXT:    [[TMP0:%.*]] = load i8, i8* [[__M_ADDR]], align 1
// CHECK-NEXT:    [[TMP1:%.*]] = load <8 x double>, <8 x double>* [[__W_ADDR]], align 64
// CHECK-NEXT:    store i8 [[TMP0]], i8* [[__M_ADDR_I]], align 1
// CHECK-NEXT:    store <8 x double> [[TMP1]], <8 x double>* [[__V_ADDR_I]], align 64
// CHECK-NEXT:    store double 0x7FF0000000000000, double* [[__W_ADDR_I_I]], align 8
// CHECK-NEXT:    [[TMP2:%.*]] = load double, double* [[__W_ADDR_I_I]], align 8
// CHECK-NEXT:    [[VECINIT_I_I:%.*]] = insertelement <8 x double> undef, double [[TMP2]], i32 0
// CHECK-NEXT:    [[TMP3:%.*]] = load double, double* [[__W_ADDR_I_I]], align 8
// CHECK-NEXT:    [[VECINIT1_I_I:%.*]] = insertelement <8 x double> [[VECINIT_I_I]], double [[TMP3]], i32 1
// CHECK-NEXT:    [[TMP4:%.*]] = load double, double* [[__W_ADDR_I_I]], align 8
// CHECK-NEXT:    [[VECINIT2_I_I:%.*]] = insertelement <8 x double> [[VECINIT1_I_I]], double [[TMP4]], i32 2
// CHECK-NEXT:    [[TMP5:%.*]] = load double, double* [[__W_ADDR_I_I]], align 8
// CHECK-NEXT:    [[VECINIT3_I_I:%.*]] = insertelement <8 x double> [[VECINIT2_I_I]], double [[TMP5]], i32 3
// CHECK-NEXT:    [[TMP6:%.*]] = load double, double* [[__W_ADDR_I_I]], align 8
// CHECK-NEXT:    [[VECINIT4_I_I:%.*]] = insertelement <8 x double> [[VECINIT3_I_I]], double [[TMP6]], i32 4
// CHECK-NEXT:    [[TMP7:%.*]] = load double, double* [[__W_ADDR_I_I]], align 8
// CHECK-NEXT:    [[VECINIT5_I_I:%.*]] = insertelement <8 x double> [[VECINIT4_I_I]], double [[TMP7]], i32 5
// CHECK-NEXT:    [[TMP8:%.*]] = load double, double* [[__W_ADDR_I_I]], align 8
// CHECK-NEXT:    [[VECINIT6_I_I:%.*]] = insertelement <8 x double> [[VECINIT5_I_I]], double [[TMP8]], i32 6
// CHECK-NEXT:    [[TMP9:%.*]] = load double, double* [[__W_ADDR_I_I]], align 8
// CHECK-NEXT:    [[VECINIT7_I_I:%.*]] = insertelement <8 x double> [[VECINIT6_I_I]], double [[TMP9]], i32 7
// CHECK-NEXT:    store <8 x double> [[VECINIT7_I_I]], <8 x double>* [[DOTCOMPOUNDLITERAL_I_I]], align 64
// CHECK-NEXT:    [[TMP10:%.*]] = load <8 x double>, <8 x double>* [[DOTCOMPOUNDLITERAL_I_I]], align 64
// CHECK-NEXT:    [[TMP11:%.*]] = load i8, i8* [[__M_ADDR_I]], align 1
// CHECK-NEXT:    [[TMP12:%.*]] = load <8 x double>, <8 x double>* [[__V_ADDR_I]], align 64
// CHECK-NEXT:    store <8 x double> [[TMP10]], <8 x double>* [[__W2_ADDR_I_I]], align 64
// CHECK-NEXT:    store i8 [[TMP11]], i8* [[__U_ADDR_I_I]], align 1
// CHECK-NEXT:    store <8 x double> [[TMP12]], <8 x double>* [[__A_ADDR_I_I]], align 64
// CHECK-NEXT:    [[TMP13:%.*]] = load i8, i8* [[__U_ADDR_I_I]], align 1
// CHECK-NEXT:    [[TMP14:%.*]] = load <8 x double>, <8 x double>* [[__A_ADDR_I_I]], align 64
// CHECK-NEXT:    [[TMP15:%.*]] = load <8 x double>, <8 x double>* [[__W2_ADDR_I_I]], align 64
// CHECK-NEXT:    [[TMP16:%.*]] = bitcast i8 [[TMP13]] to <8 x i1>
// CHECK-NEXT:    [[TMP17:%.*]] = select <8 x i1> [[TMP16]], <8 x double> [[TMP14]], <8 x double> [[TMP15]]
// CHECK-NEXT:    store <8 x double> [[TMP17]], <8 x double>* [[__V_ADDR_I]], align 64
// CHECK-NEXT:    [[TMP18:%.*]] = load <8 x double>, <8 x double>* [[__V_ADDR_I]], align 64
// CHECK-NEXT:    [[EXTRACT_I:%.*]] = shufflevector <8 x double> [[TMP18]], <8 x double> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// CHECK-NEXT:    store <4 x double> [[EXTRACT_I]], <4 x double>* [[__T1_I]], align 32
// CHECK-NEXT:    [[TMP19:%.*]] = load <8 x double>, <8 x double>* [[__V_ADDR_I]], align 64
// CHECK-NEXT:    [[EXTRACT4_I:%.*]] = shufflevector <8 x double> [[TMP19]], <8 x double> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK-NEXT:    store <4 x double> [[EXTRACT4_I]], <4 x double>* [[__T2_I]], align 32
// CHECK-NEXT:    [[TMP20:%.*]] = load <4 x double>, <4 x double>* [[__T1_I]], align 32
// CHECK-NEXT:    [[TMP21:%.*]] = load <4 x double>, <4 x double>* [[__T2_I]], align 32
// CHECK-NEXT:    store <4 x double> [[TMP20]], <4 x double>* [[__A_ADDR_I12_I]], align 32
// CHECK-NEXT:    store <4 x double> [[TMP21]], <4 x double>* [[__B_ADDR_I13_I]], align 32
// CHECK-NEXT:    [[TMP22:%.*]] = load <4 x double>, <4 x double>* [[__A_ADDR_I12_I]], align 32
// CHECK-NEXT:    [[TMP23:%.*]] = load <4 x double>, <4 x double>* [[__B_ADDR_I13_I]], align 32
// CHECK-NEXT:    [[TMP24:%.*]] = call <4 x double> @llvm.x86.avx.min.pd.256(<4 x double> [[TMP22]], <4 x double> [[TMP23]]) #2
// CHECK-NEXT:    store <4 x double> [[TMP24]], <4 x double>* [[__T3_I]], align 32
// CHECK-NEXT:    [[TMP25:%.*]] = load <4 x double>, <4 x double>* [[__T3_I]], align 32
// CHECK-NEXT:    [[EXTRACT6_I:%.*]] = shufflevector <4 x double> [[TMP25]], <4 x double> undef, <2 x i32> <i32 0, i32 1>
// CHECK-NEXT:    store <2 x double> [[EXTRACT6_I]], <2 x double>* [[__T4_I]], align 16
// CHECK-NEXT:    [[TMP26:%.*]] = load <4 x double>, <4 x double>* [[__T3_I]], align 32
// CHECK-NEXT:    [[EXTRACT7_I:%.*]] = shufflevector <4 x double> [[TMP26]], <4 x double> undef, <2 x i32> <i32 2, i32 3>
// CHECK-NEXT:    store <2 x double> [[EXTRACT7_I]], <2 x double>* [[__T5_I]], align 16
// CHECK-NEXT:    [[TMP27:%.*]] = load <2 x double>, <2 x double>* [[__T4_I]], align 16
// CHECK-NEXT:    [[TMP28:%.*]] = load <2 x double>, <2 x double>* [[__T5_I]], align 16
// CHECK-NEXT:    store <2 x double> [[TMP27]], <2 x double>* [[__A_ADDR_I10_I]], align 16
// CHECK-NEXT:    store <2 x double> [[TMP28]], <2 x double>* [[__B_ADDR_I11_I]], align 16
// CHECK-NEXT:    [[TMP29:%.*]] = load <2 x double>, <2 x double>* [[__A_ADDR_I10_I]], align 16
// CHECK-NEXT:    [[TMP30:%.*]] = load <2 x double>, <2 x double>* [[__B_ADDR_I11_I]], align 16
// CHECK-NEXT:    [[TMP31:%.*]] = call <2 x double> @llvm.x86.sse2.min.pd(<2 x double> [[TMP29]], <2 x double> [[TMP30]]) #2
// CHECK-NEXT:    store <2 x double> [[TMP31]], <2 x double>* [[__T6_I]], align 16
// CHECK-NEXT:    [[TMP32:%.*]] = load <2 x double>, <2 x double>* [[__T6_I]], align 16
// CHECK-NEXT:    [[TMP33:%.*]] = load <2 x double>, <2 x double>* [[__T6_I]], align 16
// CHECK-NEXT:    [[SHUFFLE_I:%.*]] = shufflevector <2 x double> [[TMP32]], <2 x double> [[TMP33]], <2 x i32> <i32 1, i32 0>
// CHECK-NEXT:    store <2 x double> [[SHUFFLE_I]], <2 x double>* [[__T7_I]], align 16
// CHECK-NEXT:    [[TMP34:%.*]] = load <2 x double>, <2 x double>* [[__T6_I]], align 16
// CHECK-NEXT:    [[TMP35:%.*]] = load <2 x double>, <2 x double>* [[__T7_I]], align 16
// CHECK-NEXT:    store <2 x double> [[TMP34]], <2 x double>* [[__A2_ADDR_I_I]], align 16
// CHECK-NEXT:    store <2 x double> [[TMP35]], <2 x double>* [[__B_ADDR_I_I]], align 16
// CHECK-NEXT:    [[TMP36:%.*]] = load <2 x double>, <2 x double>* [[__A2_ADDR_I_I]], align 16
// CHECK-NEXT:    [[TMP37:%.*]] = load <2 x double>, <2 x double>* [[__B_ADDR_I_I]], align 16
// CHECK-NEXT:    [[TMP38:%.*]] = call <2 x double> @llvm.x86.sse2.min.pd(<2 x double> [[TMP36]], <2 x double> [[TMP37]]) #2
// CHECK-NEXT:    store <2 x double> [[TMP38]], <2 x double>* [[__T8_I]], align 16
// CHECK-NEXT:    [[TMP39:%.*]] = load <2 x double>, <2 x double>* [[__T8_I]], align 16
// CHECK-NEXT:    [[VECEXT_I:%.*]] = extractelement <2 x double> [[TMP39]], i32 0
// CHECK-NEXT:    ret double [[VECEXT_I]]
double test_mm512_mask_reduce_min_pd(__mmask8 __M, __m512d __W){
  return _mm512_mask_reduce_min_pd(__M, __W); 
}

// CHECK-LABEL: define i32 @test_mm512_reduce_max_epi32(<8 x i64> %__W) #0 {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[__A_ADDR_I_I:%.*]] = alloca <4 x i64>, align 32
// CHECK-NEXT:    [[__B_ADDR_I_I:%.*]] = alloca <4 x i64>, align 32
// CHECK-NEXT:    [[__V1_ADDR_I12_I:%.*]] = alloca <2 x i64>, align 16
// CHECK-NEXT:    [[__V2_ADDR_I13_I:%.*]] = alloca <2 x i64>, align 16
// CHECK-NEXT:    [[__V1_ADDR_I10_I:%.*]] = alloca <2 x i64>, align 16
// CHECK-NEXT:    [[__V2_ADDR_I11_I:%.*]] = alloca <2 x i64>, align 16
// CHECK-NEXT:    [[__V1_ADDR_I_I:%.*]] = alloca <2 x i64>, align 16
// CHECK-NEXT:    [[__V2_ADDR_I_I:%.*]] = alloca <2 x i64>, align 16
// CHECK-NEXT:    [[__V_ADDR_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__T1_I:%.*]] = alloca <4 x i64>, align 32
// CHECK-NEXT:    [[__T2_I:%.*]] = alloca <4 x i64>, align 32
// CHECK-NEXT:    [[__T3_I:%.*]] = alloca <4 x i64>, align 32
// CHECK-NEXT:    [[__T4_I:%.*]] = alloca <2 x i64>, align 16
// CHECK-NEXT:    [[__T5_I:%.*]] = alloca <2 x i64>, align 16
// CHECK-NEXT:    [[__T6_I:%.*]] = alloca <2 x i64>, align 16
// CHECK-NEXT:    [[__T7_I:%.*]] = alloca <2 x i64>, align 16
// CHECK-NEXT:    [[__T8_I:%.*]] = alloca <2 x i64>, align 16
// CHECK-NEXT:    [[__T9_I:%.*]] = alloca <2 x i64>, align 16
// CHECK-NEXT:    [[__T10_I:%.*]] = alloca <4 x i32>, align 16
// CHECK-NEXT:    [[__W_ADDR:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    store <8 x i64> [[__W:%.*]], <8 x i64>* [[__W_ADDR]], align 64
// CHECK-NEXT:    [[TMP0:%.*]] = load <8 x i64>, <8 x i64>* [[__W_ADDR]], align 64
// CHECK-NEXT:    store <8 x i64> [[TMP0]], <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK-NEXT:    [[TMP1:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK-NEXT:    [[EXTRACT_I:%.*]] = shufflevector <8 x i64> [[TMP1]], <8 x i64> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// CHECK-NEXT:    store <4 x i64> [[EXTRACT_I]], <4 x i64>* [[__T1_I]], align 32
// CHECK-NEXT:    [[TMP2:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK-NEXT:    [[EXTRACT2_I:%.*]] = shufflevector <8 x i64> [[TMP2]], <8 x i64> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK-NEXT:    store <4 x i64> [[EXTRACT2_I]], <4 x i64>* [[__T2_I]], align 32
// CHECK-NEXT:    [[TMP3:%.*]] = load <4 x i64>, <4 x i64>* [[__T1_I]], align 32
// CHECK-NEXT:    [[TMP4:%.*]] = load <4 x i64>, <4 x i64>* [[__T2_I]], align 32
// CHECK-NEXT:    store <4 x i64> [[TMP3]], <4 x i64>* [[__A_ADDR_I_I]], align 32
// CHECK-NEXT:    store <4 x i64> [[TMP4]], <4 x i64>* [[__B_ADDR_I_I]], align 32
// CHECK-NEXT:    [[TMP5:%.*]] = load <4 x i64>, <4 x i64>* [[__A_ADDR_I_I]], align 32
// CHECK-NEXT:    [[TMP6:%.*]] = bitcast <4 x i64> [[TMP5]] to <8 x i32>
// CHECK-NEXT:    [[TMP7:%.*]] = load <4 x i64>, <4 x i64>* [[__B_ADDR_I_I]], align 32
// CHECK-NEXT:    [[TMP8:%.*]] = bitcast <4 x i64> [[TMP7]] to <8 x i32>
// CHECK-NEXT:    [[TMP9:%.*]] = icmp sgt <8 x i32> [[TMP6]], [[TMP8]]
// CHECK-NEXT:    [[TMP10:%.*]] = select <8 x i1> [[TMP9]], <8 x i32> [[TMP6]], <8 x i32> [[TMP8]]
// CHECK-NEXT:    [[TMP11:%.*]] = bitcast <8 x i32> [[TMP10]] to <4 x i64>
// CHECK-NEXT:    store <4 x i64> [[TMP11]], <4 x i64>* [[__T3_I]], align 32
// CHECK-NEXT:    [[TMP12:%.*]] = load <4 x i64>, <4 x i64>* [[__T3_I]], align 32
// CHECK-NEXT:    [[EXTRACT4_I:%.*]] = shufflevector <4 x i64> [[TMP12]], <4 x i64> undef, <2 x i32> <i32 0, i32 1>
// CHECK-NEXT:    store <2 x i64> [[EXTRACT4_I]], <2 x i64>* [[__T4_I]], align 16
// CHECK-NEXT:    [[TMP13:%.*]] = load <4 x i64>, <4 x i64>* [[__T3_I]], align 32
// CHECK-NEXT:    [[EXTRACT5_I:%.*]] = shufflevector <4 x i64> [[TMP13]], <4 x i64> undef, <2 x i32> <i32 2, i32 3>
// CHECK-NEXT:    store <2 x i64> [[EXTRACT5_I]], <2 x i64>* [[__T5_I]], align 16
// CHECK-NEXT:    [[TMP14:%.*]] = load <2 x i64>, <2 x i64>* [[__T4_I]], align 16
// CHECK-NEXT:    [[TMP15:%.*]] = load <2 x i64>, <2 x i64>* [[__T5_I]], align 16
// CHECK-NEXT:    store <2 x i64> [[TMP14]], <2 x i64>* [[__V1_ADDR_I12_I]], align 16
// CHECK-NEXT:    store <2 x i64> [[TMP15]], <2 x i64>* [[__V2_ADDR_I13_I]], align 16
// CHECK-NEXT:    [[TMP16:%.*]] = load <2 x i64>, <2 x i64>* [[__V1_ADDR_I12_I]], align 16
// CHECK-NEXT:    [[TMP17:%.*]] = bitcast <2 x i64> [[TMP16]] to <4 x i32>
// CHECK-NEXT:    [[TMP18:%.*]] = load <2 x i64>, <2 x i64>* [[__V2_ADDR_I13_I]], align 16
// CHECK-NEXT:    [[TMP19:%.*]] = bitcast <2 x i64> [[TMP18]] to <4 x i32>
// CHECK-NEXT:    [[TMP20:%.*]] = icmp sgt <4 x i32> [[TMP17]], [[TMP19]]
// CHECK-NEXT:    [[TMP21:%.*]] = select <4 x i1> [[TMP20]], <4 x i32> [[TMP17]], <4 x i32> [[TMP19]]
// CHECK-NEXT:    [[TMP22:%.*]] = bitcast <4 x i32> [[TMP21]] to <2 x i64>
// CHECK-NEXT:    store <2 x i64> [[TMP22]], <2 x i64>* [[__T6_I]], align 16
// CHECK-NEXT:    [[TMP23:%.*]] = load <2 x i64>, <2 x i64>* [[__T6_I]], align 16
// CHECK-NEXT:    [[TMP24:%.*]] = bitcast <2 x i64> [[TMP23]] to <4 x i32>
// CHECK-NEXT:    [[TMP25:%.*]] = load <2 x i64>, <2 x i64>* [[__T6_I]], align 16
// CHECK-NEXT:    [[TMP26:%.*]] = bitcast <2 x i64> [[TMP25]] to <4 x i32>
// CHECK-NEXT:    [[SHUFFLE_I:%.*]] = shufflevector <4 x i32> [[TMP24]], <4 x i32> [[TMP26]], <4 x i32> <i32 2, i32 3, i32 0, i32 1>
// CHECK-NEXT:    [[TMP27:%.*]] = bitcast <4 x i32> [[SHUFFLE_I]] to <2 x i64>
// CHECK-NEXT:    store <2 x i64> [[TMP27]], <2 x i64>* [[__T7_I]], align 16
// CHECK-NEXT:    [[TMP28:%.*]] = load <2 x i64>, <2 x i64>* [[__T6_I]], align 16
// CHECK-NEXT:    [[TMP29:%.*]] = load <2 x i64>, <2 x i64>* [[__T7_I]], align 16
// CHECK-NEXT:    store <2 x i64> [[TMP28]], <2 x i64>* [[__V1_ADDR_I10_I]], align 16
// CHECK-NEXT:    store <2 x i64> [[TMP29]], <2 x i64>* [[__V2_ADDR_I11_I]], align 16
// CHECK-NEXT:    [[TMP30:%.*]] = load <2 x i64>, <2 x i64>* [[__V1_ADDR_I10_I]], align 16
// CHECK-NEXT:    [[TMP31:%.*]] = bitcast <2 x i64> [[TMP30]] to <4 x i32>
// CHECK-NEXT:    [[TMP32:%.*]] = load <2 x i64>, <2 x i64>* [[__V2_ADDR_I11_I]], align 16
// CHECK-NEXT:    [[TMP33:%.*]] = bitcast <2 x i64> [[TMP32]] to <4 x i32>
// CHECK-NEXT:    [[TMP34:%.*]] = icmp sgt <4 x i32> [[TMP31]], [[TMP33]]
// CHECK-NEXT:    [[TMP35:%.*]] = select <4 x i1> [[TMP34]], <4 x i32> [[TMP31]], <4 x i32> [[TMP33]]
// CHECK-NEXT:    [[TMP36:%.*]] = bitcast <4 x i32> [[TMP35]] to <2 x i64>
// CHECK-NEXT:    store <2 x i64> [[TMP36]], <2 x i64>* [[__T8_I]], align 16
// CHECK-NEXT:    [[TMP37:%.*]] = load <2 x i64>, <2 x i64>* [[__T8_I]], align 16
// CHECK-NEXT:    [[TMP38:%.*]] = bitcast <2 x i64> [[TMP37]] to <4 x i32>
// CHECK-NEXT:    [[TMP39:%.*]] = load <2 x i64>, <2 x i64>* [[__T8_I]], align 16
// CHECK-NEXT:    [[TMP40:%.*]] = bitcast <2 x i64> [[TMP39]] to <4 x i32>
// CHECK-NEXT:    [[SHUFFLE8_I:%.*]] = shufflevector <4 x i32> [[TMP38]], <4 x i32> [[TMP40]], <4 x i32> <i32 1, i32 0, i32 3, i32 2>
// CHECK-NEXT:    [[TMP41:%.*]] = bitcast <4 x i32> [[SHUFFLE8_I]] to <2 x i64>
// CHECK-NEXT:    store <2 x i64> [[TMP41]], <2 x i64>* [[__T9_I]], align 16
// CHECK-NEXT:    [[TMP42:%.*]] = load <2 x i64>, <2 x i64>* [[__T8_I]], align 16
// CHECK-NEXT:    [[TMP43:%.*]] = load <2 x i64>, <2 x i64>* [[__T9_I]], align 16
// CHECK-NEXT:    store <2 x i64> [[TMP42]], <2 x i64>* [[__V1_ADDR_I_I]], align 16
// CHECK-NEXT:    store <2 x i64> [[TMP43]], <2 x i64>* [[__V2_ADDR_I_I]], align 16
// CHECK-NEXT:    [[TMP44:%.*]] = load <2 x i64>, <2 x i64>* [[__V1_ADDR_I_I]], align 16
// CHECK-NEXT:    [[TMP45:%.*]] = bitcast <2 x i64> [[TMP44]] to <4 x i32>
// CHECK-NEXT:    [[TMP46:%.*]] = load <2 x i64>, <2 x i64>* [[__V2_ADDR_I_I]], align 16
// CHECK-NEXT:    [[TMP47:%.*]] = bitcast <2 x i64> [[TMP46]] to <4 x i32>
// CHECK-NEXT:    [[TMP48:%.*]] = icmp sgt <4 x i32> [[TMP45]], [[TMP47]]
// CHECK-NEXT:    [[TMP49:%.*]] = select <4 x i1> [[TMP48]], <4 x i32> [[TMP45]], <4 x i32> [[TMP47]]
// CHECK-NEXT:    [[TMP50:%.*]] = bitcast <4 x i32> [[TMP49]] to <2 x i64>
// CHECK-NEXT:    store <4 x i32> [[TMP49]], <4 x i32>* [[__T10_I]], align 16
// CHECK-NEXT:    [[TMP51:%.*]] = load <4 x i32>, <4 x i32>* [[__T10_I]], align 16
// CHECK-NEXT:    [[VECEXT_I:%.*]] = extractelement <4 x i32> [[TMP51]], i32 0
// CHECK-NEXT:    ret i32 [[VECEXT_I]]
int test_mm512_reduce_max_epi32(__m512i __W){
  return _mm512_reduce_max_epi32(__W);
}

// CHECK-LABEL: define i32 @test_mm512_reduce_max_epu32(<8 x i64> %__W) #0 {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[__A_ADDR_I_I:%.*]] = alloca <4 x i64>, align 32
// CHECK-NEXT:    [[__B_ADDR_I_I:%.*]] = alloca <4 x i64>, align 32
// CHECK-NEXT:    [[__V1_ADDR_I12_I:%.*]] = alloca <2 x i64>, align 16
// CHECK-NEXT:    [[__V2_ADDR_I13_I:%.*]] = alloca <2 x i64>, align 16
// CHECK-NEXT:    [[__V1_ADDR_I10_I:%.*]] = alloca <2 x i64>, align 16
// CHECK-NEXT:    [[__V2_ADDR_I11_I:%.*]] = alloca <2 x i64>, align 16
// CHECK-NEXT:    [[__V1_ADDR_I_I:%.*]] = alloca <2 x i64>, align 16
// CHECK-NEXT:    [[__V2_ADDR_I_I:%.*]] = alloca <2 x i64>, align 16
// CHECK-NEXT:    [[__V_ADDR_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__T1_I:%.*]] = alloca <4 x i64>, align 32
// CHECK-NEXT:    [[__T2_I:%.*]] = alloca <4 x i64>, align 32
// CHECK-NEXT:    [[__T3_I:%.*]] = alloca <4 x i64>, align 32
// CHECK-NEXT:    [[__T4_I:%.*]] = alloca <2 x i64>, align 16
// CHECK-NEXT:    [[__T5_I:%.*]] = alloca <2 x i64>, align 16
// CHECK-NEXT:    [[__T6_I:%.*]] = alloca <2 x i64>, align 16
// CHECK-NEXT:    [[__T7_I:%.*]] = alloca <2 x i64>, align 16
// CHECK-NEXT:    [[__T8_I:%.*]] = alloca <2 x i64>, align 16
// CHECK-NEXT:    [[__T9_I:%.*]] = alloca <2 x i64>, align 16
// CHECK-NEXT:    [[__T10_I:%.*]] = alloca <4 x i32>, align 16
// CHECK-NEXT:    [[__W_ADDR:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    store <8 x i64> [[__W:%.*]], <8 x i64>* [[__W_ADDR]], align 64
// CHECK-NEXT:    [[TMP0:%.*]] = load <8 x i64>, <8 x i64>* [[__W_ADDR]], align 64
// CHECK-NEXT:    store <8 x i64> [[TMP0]], <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK-NEXT:    [[TMP1:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK-NEXT:    [[EXTRACT_I:%.*]] = shufflevector <8 x i64> [[TMP1]], <8 x i64> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// CHECK-NEXT:    store <4 x i64> [[EXTRACT_I]], <4 x i64>* [[__T1_I]], align 32
// CHECK-NEXT:    [[TMP2:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK-NEXT:    [[EXTRACT2_I:%.*]] = shufflevector <8 x i64> [[TMP2]], <8 x i64> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK-NEXT:    store <4 x i64> [[EXTRACT2_I]], <4 x i64>* [[__T2_I]], align 32
// CHECK-NEXT:    [[TMP3:%.*]] = load <4 x i64>, <4 x i64>* [[__T1_I]], align 32
// CHECK-NEXT:    [[TMP4:%.*]] = load <4 x i64>, <4 x i64>* [[__T2_I]], align 32
// CHECK-NEXT:    store <4 x i64> [[TMP3]], <4 x i64>* [[__A_ADDR_I_I]], align 32
// CHECK-NEXT:    store <4 x i64> [[TMP4]], <4 x i64>* [[__B_ADDR_I_I]], align 32
// CHECK-NEXT:    [[TMP5:%.*]] = load <4 x i64>, <4 x i64>* [[__A_ADDR_I_I]], align 32
// CHECK-NEXT:    [[TMP6:%.*]] = bitcast <4 x i64> [[TMP5]] to <8 x i32>
// CHECK-NEXT:    [[TMP7:%.*]] = load <4 x i64>, <4 x i64>* [[__B_ADDR_I_I]], align 32
// CHECK-NEXT:    [[TMP8:%.*]] = bitcast <4 x i64> [[TMP7]] to <8 x i32>
// CHECK-NEXT:    [[TMP9:%.*]] = icmp ugt <8 x i32> [[TMP6]], [[TMP8]]
// CHECK-NEXT:    [[TMP10:%.*]] = select <8 x i1> [[TMP9]], <8 x i32> [[TMP6]], <8 x i32> [[TMP8]]
// CHECK-NEXT:    [[TMP11:%.*]] = bitcast <8 x i32> [[TMP10]] to <4 x i64>
// CHECK-NEXT:    store <4 x i64> [[TMP11]], <4 x i64>* [[__T3_I]], align 32
// CHECK-NEXT:    [[TMP12:%.*]] = load <4 x i64>, <4 x i64>* [[__T3_I]], align 32
// CHECK-NEXT:    [[EXTRACT4_I:%.*]] = shufflevector <4 x i64> [[TMP12]], <4 x i64> undef, <2 x i32> <i32 0, i32 1>
// CHECK-NEXT:    store <2 x i64> [[EXTRACT4_I]], <2 x i64>* [[__T4_I]], align 16
// CHECK-NEXT:    [[TMP13:%.*]] = load <4 x i64>, <4 x i64>* [[__T3_I]], align 32
// CHECK-NEXT:    [[EXTRACT5_I:%.*]] = shufflevector <4 x i64> [[TMP13]], <4 x i64> undef, <2 x i32> <i32 2, i32 3>
// CHECK-NEXT:    store <2 x i64> [[EXTRACT5_I]], <2 x i64>* [[__T5_I]], align 16
// CHECK-NEXT:    [[TMP14:%.*]] = load <2 x i64>, <2 x i64>* [[__T4_I]], align 16
// CHECK-NEXT:    [[TMP15:%.*]] = load <2 x i64>, <2 x i64>* [[__T5_I]], align 16
// CHECK-NEXT:    store <2 x i64> [[TMP14]], <2 x i64>* [[__V1_ADDR_I12_I]], align 16
// CHECK-NEXT:    store <2 x i64> [[TMP15]], <2 x i64>* [[__V2_ADDR_I13_I]], align 16
// CHECK-NEXT:    [[TMP16:%.*]] = load <2 x i64>, <2 x i64>* [[__V1_ADDR_I12_I]], align 16
// CHECK-NEXT:    [[TMP17:%.*]] = bitcast <2 x i64> [[TMP16]] to <4 x i32>
// CHECK-NEXT:    [[TMP18:%.*]] = load <2 x i64>, <2 x i64>* [[__V2_ADDR_I13_I]], align 16
// CHECK-NEXT:    [[TMP19:%.*]] = bitcast <2 x i64> [[TMP18]] to <4 x i32>
// CHECK-NEXT:    [[TMP20:%.*]] = icmp ugt <4 x i32> [[TMP17]], [[TMP19]]
// CHECK-NEXT:    [[TMP21:%.*]] = select <4 x i1> [[TMP20]], <4 x i32> [[TMP17]], <4 x i32> [[TMP19]]
// CHECK-NEXT:    [[TMP22:%.*]] = bitcast <4 x i32> [[TMP21]] to <2 x i64>
// CHECK-NEXT:    store <2 x i64> [[TMP22]], <2 x i64>* [[__T6_I]], align 16
// CHECK-NEXT:    [[TMP23:%.*]] = load <2 x i64>, <2 x i64>* [[__T6_I]], align 16
// CHECK-NEXT:    [[TMP24:%.*]] = bitcast <2 x i64> [[TMP23]] to <4 x i32>
// CHECK-NEXT:    [[TMP25:%.*]] = load <2 x i64>, <2 x i64>* [[__T6_I]], align 16
// CHECK-NEXT:    [[TMP26:%.*]] = bitcast <2 x i64> [[TMP25]] to <4 x i32>
// CHECK-NEXT:    [[SHUFFLE_I:%.*]] = shufflevector <4 x i32> [[TMP24]], <4 x i32> [[TMP26]], <4 x i32> <i32 2, i32 3, i32 0, i32 1>
// CHECK-NEXT:    [[TMP27:%.*]] = bitcast <4 x i32> [[SHUFFLE_I]] to <2 x i64>
// CHECK-NEXT:    store <2 x i64> [[TMP27]], <2 x i64>* [[__T7_I]], align 16
// CHECK-NEXT:    [[TMP28:%.*]] = load <2 x i64>, <2 x i64>* [[__T6_I]], align 16
// CHECK-NEXT:    [[TMP29:%.*]] = load <2 x i64>, <2 x i64>* [[__T7_I]], align 16
// CHECK-NEXT:    store <2 x i64> [[TMP28]], <2 x i64>* [[__V1_ADDR_I10_I]], align 16
// CHECK-NEXT:    store <2 x i64> [[TMP29]], <2 x i64>* [[__V2_ADDR_I11_I]], align 16
// CHECK-NEXT:    [[TMP30:%.*]] = load <2 x i64>, <2 x i64>* [[__V1_ADDR_I10_I]], align 16
// CHECK-NEXT:    [[TMP31:%.*]] = bitcast <2 x i64> [[TMP30]] to <4 x i32>
// CHECK-NEXT:    [[TMP32:%.*]] = load <2 x i64>, <2 x i64>* [[__V2_ADDR_I11_I]], align 16
// CHECK-NEXT:    [[TMP33:%.*]] = bitcast <2 x i64> [[TMP32]] to <4 x i32>
// CHECK-NEXT:    [[TMP34:%.*]] = icmp ugt <4 x i32> [[TMP31]], [[TMP33]]
// CHECK-NEXT:    [[TMP35:%.*]] = select <4 x i1> [[TMP34]], <4 x i32> [[TMP31]], <4 x i32> [[TMP33]]
// CHECK-NEXT:    [[TMP36:%.*]] = bitcast <4 x i32> [[TMP35]] to <2 x i64>
// CHECK-NEXT:    store <2 x i64> [[TMP36]], <2 x i64>* [[__T8_I]], align 16
// CHECK-NEXT:    [[TMP37:%.*]] = load <2 x i64>, <2 x i64>* [[__T8_I]], align 16
// CHECK-NEXT:    [[TMP38:%.*]] = bitcast <2 x i64> [[TMP37]] to <4 x i32>
// CHECK-NEXT:    [[TMP39:%.*]] = load <2 x i64>, <2 x i64>* [[__T8_I]], align 16
// CHECK-NEXT:    [[TMP40:%.*]] = bitcast <2 x i64> [[TMP39]] to <4 x i32>
// CHECK-NEXT:    [[SHUFFLE8_I:%.*]] = shufflevector <4 x i32> [[TMP38]], <4 x i32> [[TMP40]], <4 x i32> <i32 1, i32 0, i32 3, i32 2>
// CHECK-NEXT:    [[TMP41:%.*]] = bitcast <4 x i32> [[SHUFFLE8_I]] to <2 x i64>
// CHECK-NEXT:    store <2 x i64> [[TMP41]], <2 x i64>* [[__T9_I]], align 16
// CHECK-NEXT:    [[TMP42:%.*]] = load <2 x i64>, <2 x i64>* [[__T8_I]], align 16
// CHECK-NEXT:    [[TMP43:%.*]] = load <2 x i64>, <2 x i64>* [[__T9_I]], align 16
// CHECK-NEXT:    store <2 x i64> [[TMP42]], <2 x i64>* [[__V1_ADDR_I_I]], align 16
// CHECK-NEXT:    store <2 x i64> [[TMP43]], <2 x i64>* [[__V2_ADDR_I_I]], align 16
// CHECK-NEXT:    [[TMP44:%.*]] = load <2 x i64>, <2 x i64>* [[__V1_ADDR_I_I]], align 16
// CHECK-NEXT:    [[TMP45:%.*]] = bitcast <2 x i64> [[TMP44]] to <4 x i32>
// CHECK-NEXT:    [[TMP46:%.*]] = load <2 x i64>, <2 x i64>* [[__V2_ADDR_I_I]], align 16
// CHECK-NEXT:    [[TMP47:%.*]] = bitcast <2 x i64> [[TMP46]] to <4 x i32>
// CHECK-NEXT:    [[TMP48:%.*]] = icmp ugt <4 x i32> [[TMP45]], [[TMP47]]
// CHECK-NEXT:    [[TMP49:%.*]] = select <4 x i1> [[TMP48]], <4 x i32> [[TMP45]], <4 x i32> [[TMP47]]
// CHECK-NEXT:    [[TMP50:%.*]] = bitcast <4 x i32> [[TMP49]] to <2 x i64>
// CHECK-NEXT:    store <4 x i32> [[TMP49]], <4 x i32>* [[__T10_I]], align 16
// CHECK-NEXT:    [[TMP51:%.*]] = load <4 x i32>, <4 x i32>* [[__T10_I]], align 16
// CHECK-NEXT:    [[VECEXT_I:%.*]] = extractelement <4 x i32> [[TMP51]], i32 0
// CHECK-NEXT:    ret i32 [[VECEXT_I]]
unsigned int test_mm512_reduce_max_epu32(__m512i __W){
  return _mm512_reduce_max_epu32(__W); 
}

// CHECK-LABEL: define float @test_mm512_reduce_max_ps(<16 x float> %__W) #0 {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[__A_ADDR_I14_I:%.*]] = alloca <8 x float>, align 32
// CHECK-NEXT:    [[__B_ADDR_I15_I:%.*]] = alloca <8 x float>, align 32
// CHECK-NEXT:    [[__A_ADDR_I12_I:%.*]] = alloca <4 x float>, align 16
// CHECK-NEXT:    [[__B_ADDR_I13_I:%.*]] = alloca <4 x float>, align 16
// CHECK-NEXT:    [[__A_ADDR_I10_I:%.*]] = alloca <4 x float>, align 16
// CHECK-NEXT:    [[__B_ADDR_I11_I:%.*]] = alloca <4 x float>, align 16
// CHECK-NEXT:    [[__A_ADDR_I_I:%.*]] = alloca <4 x float>, align 16
// CHECK-NEXT:    [[__B_ADDR_I_I:%.*]] = alloca <4 x float>, align 16
// CHECK-NEXT:    [[__V_ADDR_I:%.*]] = alloca <16 x float>, align 64
// CHECK-NEXT:    [[__T1_I:%.*]] = alloca <8 x float>, align 32
// CHECK-NEXT:    [[__T2_I:%.*]] = alloca <8 x float>, align 32
// CHECK-NEXT:    [[__T3_I:%.*]] = alloca <8 x float>, align 32
// CHECK-NEXT:    [[__T4_I:%.*]] = alloca <4 x float>, align 16
// CHECK-NEXT:    [[__T5_I:%.*]] = alloca <4 x float>, align 16
// CHECK-NEXT:    [[__T6_I:%.*]] = alloca <4 x float>, align 16
// CHECK-NEXT:    [[__T7_I:%.*]] = alloca <4 x float>, align 16
// CHECK-NEXT:    [[__T8_I:%.*]] = alloca <4 x float>, align 16
// CHECK-NEXT:    [[__T9_I:%.*]] = alloca <4 x float>, align 16
// CHECK-NEXT:    [[__T10_I:%.*]] = alloca <4 x float>, align 16
// CHECK-NEXT:    [[__W_ADDR:%.*]] = alloca <16 x float>, align 64
// CHECK-NEXT:    store <16 x float> [[__W:%.*]], <16 x float>* [[__W_ADDR]], align 64
// CHECK-NEXT:    [[TMP0:%.*]] = load <16 x float>, <16 x float>* [[__W_ADDR]], align 64
// CHECK-NEXT:    store <16 x float> [[TMP0]], <16 x float>* [[__V_ADDR_I]], align 64
// CHECK-NEXT:    [[TMP1:%.*]] = load <16 x float>, <16 x float>* [[__V_ADDR_I]], align 64
// CHECK-NEXT:    [[TMP2:%.*]] = bitcast <16 x float> [[TMP1]] to <8 x double>
// CHECK-NEXT:    [[EXTRACT_I:%.*]] = shufflevector <8 x double> [[TMP2]], <8 x double> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// CHECK-NEXT:    [[TMP3:%.*]] = bitcast <4 x double> [[EXTRACT_I]] to <8 x float>
// CHECK-NEXT:    store <8 x float> [[TMP3]], <8 x float>* [[__T1_I]], align 32
// CHECK-NEXT:    [[TMP4:%.*]] = load <16 x float>, <16 x float>* [[__V_ADDR_I]], align 64
// CHECK-NEXT:    [[TMP5:%.*]] = bitcast <16 x float> [[TMP4]] to <8 x double>
// CHECK-NEXT:    [[EXTRACT2_I:%.*]] = shufflevector <8 x double> [[TMP5]], <8 x double> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK-NEXT:    [[TMP6:%.*]] = bitcast <4 x double> [[EXTRACT2_I]] to <8 x float>
// CHECK-NEXT:    store <8 x float> [[TMP6]], <8 x float>* [[__T2_I]], align 32
// CHECK-NEXT:    [[TMP7:%.*]] = load <8 x float>, <8 x float>* [[__T1_I]], align 32
// CHECK-NEXT:    [[TMP8:%.*]] = load <8 x float>, <8 x float>* [[__T2_I]], align 32
// CHECK-NEXT:    store <8 x float> [[TMP7]], <8 x float>* [[__A_ADDR_I14_I]], align 32
// CHECK-NEXT:    store <8 x float> [[TMP8]], <8 x float>* [[__B_ADDR_I15_I]], align 32
// CHECK-NEXT:    [[TMP9:%.*]] = load <8 x float>, <8 x float>* [[__A_ADDR_I14_I]], align 32
// CHECK-NEXT:    [[TMP10:%.*]] = load <8 x float>, <8 x float>* [[__B_ADDR_I15_I]], align 32
// CHECK-NEXT:    [[TMP11:%.*]] = call <8 x float> @llvm.x86.avx.max.ps.256(<8 x float> [[TMP9]], <8 x float> [[TMP10]]) #2
// CHECK-NEXT:    store <8 x float> [[TMP11]], <8 x float>* [[__T3_I]], align 32
// CHECK-NEXT:    [[TMP12:%.*]] = load <8 x float>, <8 x float>* [[__T3_I]], align 32
// CHECK-NEXT:    [[EXTRACT4_I:%.*]] = shufflevector <8 x float> [[TMP12]], <8 x float> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// CHECK-NEXT:    store <4 x float> [[EXTRACT4_I]], <4 x float>* [[__T4_I]], align 16
// CHECK-NEXT:    [[TMP13:%.*]] = load <8 x float>, <8 x float>* [[__T3_I]], align 32
// CHECK-NEXT:    [[EXTRACT5_I:%.*]] = shufflevector <8 x float> [[TMP13]], <8 x float> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK-NEXT:    store <4 x float> [[EXTRACT5_I]], <4 x float>* [[__T5_I]], align 16
// CHECK-NEXT:    [[TMP14:%.*]] = load <4 x float>, <4 x float>* [[__T4_I]], align 16
// CHECK-NEXT:    [[TMP15:%.*]] = load <4 x float>, <4 x float>* [[__T5_I]], align 16
// CHECK-NEXT:    store <4 x float> [[TMP14]], <4 x float>* [[__A_ADDR_I12_I]], align 16
// CHECK-NEXT:    store <4 x float> [[TMP15]], <4 x float>* [[__B_ADDR_I13_I]], align 16
// CHECK-NEXT:    [[TMP16:%.*]] = load <4 x float>, <4 x float>* [[__A_ADDR_I12_I]], align 16
// CHECK-NEXT:    [[TMP17:%.*]] = load <4 x float>, <4 x float>* [[__B_ADDR_I13_I]], align 16
// CHECK-NEXT:    [[TMP18:%.*]] = call <4 x float> @llvm.x86.sse.max.ps(<4 x float> [[TMP16]], <4 x float> [[TMP17]]) #2
// CHECK-NEXT:    store <4 x float> [[TMP18]], <4 x float>* [[__T6_I]], align 16
// CHECK-NEXT:    [[TMP19:%.*]] = load <4 x float>, <4 x float>* [[__T6_I]], align 16
// CHECK-NEXT:    [[TMP20:%.*]] = load <4 x float>, <4 x float>* [[__T6_I]], align 16
// CHECK-NEXT:    [[SHUFFLE_I:%.*]] = shufflevector <4 x float> [[TMP19]], <4 x float> [[TMP20]], <4 x i32> <i32 2, i32 3, i32 0, i32 1>
// CHECK-NEXT:    store <4 x float> [[SHUFFLE_I]], <4 x float>* [[__T7_I]], align 16
// CHECK-NEXT:    [[TMP21:%.*]] = load <4 x float>, <4 x float>* [[__T6_I]], align 16
// CHECK-NEXT:    [[TMP22:%.*]] = load <4 x float>, <4 x float>* [[__T7_I]], align 16
// CHECK-NEXT:    store <4 x float> [[TMP21]], <4 x float>* [[__A_ADDR_I10_I]], align 16
// CHECK-NEXT:    store <4 x float> [[TMP22]], <4 x float>* [[__B_ADDR_I11_I]], align 16
// CHECK-NEXT:    [[TMP23:%.*]] = load <4 x float>, <4 x float>* [[__A_ADDR_I10_I]], align 16
// CHECK-NEXT:    [[TMP24:%.*]] = load <4 x float>, <4 x float>* [[__B_ADDR_I11_I]], align 16
// CHECK-NEXT:    [[TMP25:%.*]] = call <4 x float> @llvm.x86.sse.max.ps(<4 x float> [[TMP23]], <4 x float> [[TMP24]]) #2
// CHECK-NEXT:    store <4 x float> [[TMP25]], <4 x float>* [[__T8_I]], align 16
// CHECK-NEXT:    [[TMP26:%.*]] = load <4 x float>, <4 x float>* [[__T8_I]], align 16
// CHECK-NEXT:    [[TMP27:%.*]] = load <4 x float>, <4 x float>* [[__T8_I]], align 16
// CHECK-NEXT:    [[SHUFFLE8_I:%.*]] = shufflevector <4 x float> [[TMP26]], <4 x float> [[TMP27]], <4 x i32> <i32 1, i32 0, i32 3, i32 2>
// CHECK-NEXT:    store <4 x float> [[SHUFFLE8_I]], <4 x float>* [[__T9_I]], align 16
// CHECK-NEXT:    [[TMP28:%.*]] = load <4 x float>, <4 x float>* [[__T8_I]], align 16
// CHECK-NEXT:    [[TMP29:%.*]] = load <4 x float>, <4 x float>* [[__T9_I]], align 16
// CHECK-NEXT:    store <4 x float> [[TMP28]], <4 x float>* [[__A_ADDR_I_I]], align 16
// CHECK-NEXT:    store <4 x float> [[TMP29]], <4 x float>* [[__B_ADDR_I_I]], align 16
// CHECK-NEXT:    [[TMP30:%.*]] = load <4 x float>, <4 x float>* [[__A_ADDR_I_I]], align 16
// CHECK-NEXT:    [[TMP31:%.*]] = load <4 x float>, <4 x float>* [[__B_ADDR_I_I]], align 16
// CHECK-NEXT:    [[TMP32:%.*]] = call <4 x float> @llvm.x86.sse.max.ps(<4 x float> [[TMP30]], <4 x float> [[TMP31]]) #2
// CHECK-NEXT:    store <4 x float> [[TMP32]], <4 x float>* [[__T10_I]], align 16
// CHECK-NEXT:    [[TMP33:%.*]] = load <4 x float>, <4 x float>* [[__T10_I]], align 16
// CHECK-NEXT:    [[VECEXT_I:%.*]] = extractelement <4 x float> [[TMP33]], i32 0
// CHECK-NEXT:    ret float [[VECEXT_I]]
float test_mm512_reduce_max_ps(__m512 __W){
  return _mm512_reduce_max_ps(__W); 
}

// CHECK-LABEL: define i32 @test_mm512_reduce_min_epi32(<8 x i64> %__W) #0 {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[__A_ADDR_I_I:%.*]] = alloca <4 x i64>, align 32
// CHECK-NEXT:    [[__B_ADDR_I_I:%.*]] = alloca <4 x i64>, align 32
// CHECK-NEXT:    [[__V1_ADDR_I12_I:%.*]] = alloca <2 x i64>, align 16
// CHECK-NEXT:    [[__V2_ADDR_I13_I:%.*]] = alloca <2 x i64>, align 16
// CHECK-NEXT:    [[__V1_ADDR_I10_I:%.*]] = alloca <2 x i64>, align 16
// CHECK-NEXT:    [[__V2_ADDR_I11_I:%.*]] = alloca <2 x i64>, align 16
// CHECK-NEXT:    [[__V1_ADDR_I_I:%.*]] = alloca <2 x i64>, align 16
// CHECK-NEXT:    [[__V2_ADDR_I_I:%.*]] = alloca <2 x i64>, align 16
// CHECK-NEXT:    [[__V_ADDR_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__T1_I:%.*]] = alloca <4 x i64>, align 32
// CHECK-NEXT:    [[__T2_I:%.*]] = alloca <4 x i64>, align 32
// CHECK-NEXT:    [[__T3_I:%.*]] = alloca <4 x i64>, align 32
// CHECK-NEXT:    [[__T4_I:%.*]] = alloca <2 x i64>, align 16
// CHECK-NEXT:    [[__T5_I:%.*]] = alloca <2 x i64>, align 16
// CHECK-NEXT:    [[__T6_I:%.*]] = alloca <2 x i64>, align 16
// CHECK-NEXT:    [[__T7_I:%.*]] = alloca <2 x i64>, align 16
// CHECK-NEXT:    [[__T8_I:%.*]] = alloca <2 x i64>, align 16
// CHECK-NEXT:    [[__T9_I:%.*]] = alloca <2 x i64>, align 16
// CHECK-NEXT:    [[__T10_I:%.*]] = alloca <4 x i32>, align 16
// CHECK-NEXT:    [[__W_ADDR:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    store <8 x i64> [[__W:%.*]], <8 x i64>* [[__W_ADDR]], align 64
// CHECK-NEXT:    [[TMP0:%.*]] = load <8 x i64>, <8 x i64>* [[__W_ADDR]], align 64
// CHECK-NEXT:    store <8 x i64> [[TMP0]], <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK-NEXT:    [[TMP1:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK-NEXT:    [[EXTRACT_I:%.*]] = shufflevector <8 x i64> [[TMP1]], <8 x i64> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// CHECK-NEXT:    store <4 x i64> [[EXTRACT_I]], <4 x i64>* [[__T1_I]], align 32
// CHECK-NEXT:    [[TMP2:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK-NEXT:    [[EXTRACT2_I:%.*]] = shufflevector <8 x i64> [[TMP2]], <8 x i64> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK-NEXT:    store <4 x i64> [[EXTRACT2_I]], <4 x i64>* [[__T2_I]], align 32
// CHECK-NEXT:    [[TMP3:%.*]] = load <4 x i64>, <4 x i64>* [[__T1_I]], align 32
// CHECK-NEXT:    [[TMP4:%.*]] = load <4 x i64>, <4 x i64>* [[__T2_I]], align 32
// CHECK-NEXT:    store <4 x i64> [[TMP3]], <4 x i64>* [[__A_ADDR_I_I]], align 32
// CHECK-NEXT:    store <4 x i64> [[TMP4]], <4 x i64>* [[__B_ADDR_I_I]], align 32
// CHECK-NEXT:    [[TMP5:%.*]] = load <4 x i64>, <4 x i64>* [[__A_ADDR_I_I]], align 32
// CHECK-NEXT:    [[TMP6:%.*]] = bitcast <4 x i64> [[TMP5]] to <8 x i32>
// CHECK-NEXT:    [[TMP7:%.*]] = load <4 x i64>, <4 x i64>* [[__B_ADDR_I_I]], align 32
// CHECK-NEXT:    [[TMP8:%.*]] = bitcast <4 x i64> [[TMP7]] to <8 x i32>
// CHECK-NEXT:    [[TMP9:%.*]] = icmp slt <8 x i32> [[TMP6]], [[TMP8]]
// CHECK-NEXT:    [[TMP10:%.*]] = select <8 x i1> [[TMP9]], <8 x i32> [[TMP6]], <8 x i32> [[TMP8]]
// CHECK-NEXT:    [[TMP11:%.*]] = bitcast <8 x i32> [[TMP10]] to <4 x i64>
// CHECK-NEXT:    store <4 x i64> [[TMP11]], <4 x i64>* [[__T3_I]], align 32
// CHECK-NEXT:    [[TMP12:%.*]] = load <4 x i64>, <4 x i64>* [[__T3_I]], align 32
// CHECK-NEXT:    [[EXTRACT4_I:%.*]] = shufflevector <4 x i64> [[TMP12]], <4 x i64> undef, <2 x i32> <i32 0, i32 1>
// CHECK-NEXT:    store <2 x i64> [[EXTRACT4_I]], <2 x i64>* [[__T4_I]], align 16
// CHECK-NEXT:    [[TMP13:%.*]] = load <4 x i64>, <4 x i64>* [[__T3_I]], align 32
// CHECK-NEXT:    [[EXTRACT5_I:%.*]] = shufflevector <4 x i64> [[TMP13]], <4 x i64> undef, <2 x i32> <i32 2, i32 3>
// CHECK-NEXT:    store <2 x i64> [[EXTRACT5_I]], <2 x i64>* [[__T5_I]], align 16
// CHECK-NEXT:    [[TMP14:%.*]] = load <2 x i64>, <2 x i64>* [[__T4_I]], align 16
// CHECK-NEXT:    [[TMP15:%.*]] = load <2 x i64>, <2 x i64>* [[__T5_I]], align 16
// CHECK-NEXT:    store <2 x i64> [[TMP14]], <2 x i64>* [[__V1_ADDR_I12_I]], align 16
// CHECK-NEXT:    store <2 x i64> [[TMP15]], <2 x i64>* [[__V2_ADDR_I13_I]], align 16
// CHECK-NEXT:    [[TMP16:%.*]] = load <2 x i64>, <2 x i64>* [[__V1_ADDR_I12_I]], align 16
// CHECK-NEXT:    [[TMP17:%.*]] = bitcast <2 x i64> [[TMP16]] to <4 x i32>
// CHECK-NEXT:    [[TMP18:%.*]] = load <2 x i64>, <2 x i64>* [[__V2_ADDR_I13_I]], align 16
// CHECK-NEXT:    [[TMP19:%.*]] = bitcast <2 x i64> [[TMP18]] to <4 x i32>
// CHECK-NEXT:    [[TMP20:%.*]] = icmp slt <4 x i32> [[TMP17]], [[TMP19]]
// CHECK-NEXT:    [[TMP21:%.*]] = select <4 x i1> [[TMP20]], <4 x i32> [[TMP17]], <4 x i32> [[TMP19]]
// CHECK-NEXT:    [[TMP22:%.*]] = bitcast <4 x i32> [[TMP21]] to <2 x i64>
// CHECK-NEXT:    store <2 x i64> [[TMP22]], <2 x i64>* [[__T6_I]], align 16
// CHECK-NEXT:    [[TMP23:%.*]] = load <2 x i64>, <2 x i64>* [[__T6_I]], align 16
// CHECK-NEXT:    [[TMP24:%.*]] = bitcast <2 x i64> [[TMP23]] to <4 x i32>
// CHECK-NEXT:    [[TMP25:%.*]] = load <2 x i64>, <2 x i64>* [[__T6_I]], align 16
// CHECK-NEXT:    [[TMP26:%.*]] = bitcast <2 x i64> [[TMP25]] to <4 x i32>
// CHECK-NEXT:    [[SHUFFLE_I:%.*]] = shufflevector <4 x i32> [[TMP24]], <4 x i32> [[TMP26]], <4 x i32> <i32 2, i32 3, i32 0, i32 1>
// CHECK-NEXT:    [[TMP27:%.*]] = bitcast <4 x i32> [[SHUFFLE_I]] to <2 x i64>
// CHECK-NEXT:    store <2 x i64> [[TMP27]], <2 x i64>* [[__T7_I]], align 16
// CHECK-NEXT:    [[TMP28:%.*]] = load <2 x i64>, <2 x i64>* [[__T6_I]], align 16
// CHECK-NEXT:    [[TMP29:%.*]] = load <2 x i64>, <2 x i64>* [[__T7_I]], align 16
// CHECK-NEXT:    store <2 x i64> [[TMP28]], <2 x i64>* [[__V1_ADDR_I10_I]], align 16
// CHECK-NEXT:    store <2 x i64> [[TMP29]], <2 x i64>* [[__V2_ADDR_I11_I]], align 16
// CHECK-NEXT:    [[TMP30:%.*]] = load <2 x i64>, <2 x i64>* [[__V1_ADDR_I10_I]], align 16
// CHECK-NEXT:    [[TMP31:%.*]] = bitcast <2 x i64> [[TMP30]] to <4 x i32>
// CHECK-NEXT:    [[TMP32:%.*]] = load <2 x i64>, <2 x i64>* [[__V2_ADDR_I11_I]], align 16
// CHECK-NEXT:    [[TMP33:%.*]] = bitcast <2 x i64> [[TMP32]] to <4 x i32>
// CHECK-NEXT:    [[TMP34:%.*]] = icmp slt <4 x i32> [[TMP31]], [[TMP33]]
// CHECK-NEXT:    [[TMP35:%.*]] = select <4 x i1> [[TMP34]], <4 x i32> [[TMP31]], <4 x i32> [[TMP33]]
// CHECK-NEXT:    [[TMP36:%.*]] = bitcast <4 x i32> [[TMP35]] to <2 x i64>
// CHECK-NEXT:    store <2 x i64> [[TMP36]], <2 x i64>* [[__T8_I]], align 16
// CHECK-NEXT:    [[TMP37:%.*]] = load <2 x i64>, <2 x i64>* [[__T8_I]], align 16
// CHECK-NEXT:    [[TMP38:%.*]] = bitcast <2 x i64> [[TMP37]] to <4 x i32>
// CHECK-NEXT:    [[TMP39:%.*]] = load <2 x i64>, <2 x i64>* [[__T8_I]], align 16
// CHECK-NEXT:    [[TMP40:%.*]] = bitcast <2 x i64> [[TMP39]] to <4 x i32>
// CHECK-NEXT:    [[SHUFFLE8_I:%.*]] = shufflevector <4 x i32> [[TMP38]], <4 x i32> [[TMP40]], <4 x i32> <i32 1, i32 0, i32 3, i32 2>
// CHECK-NEXT:    [[TMP41:%.*]] = bitcast <4 x i32> [[SHUFFLE8_I]] to <2 x i64>
// CHECK-NEXT:    store <2 x i64> [[TMP41]], <2 x i64>* [[__T9_I]], align 16
// CHECK-NEXT:    [[TMP42:%.*]] = load <2 x i64>, <2 x i64>* [[__T8_I]], align 16
// CHECK-NEXT:    [[TMP43:%.*]] = load <2 x i64>, <2 x i64>* [[__T9_I]], align 16
// CHECK-NEXT:    store <2 x i64> [[TMP42]], <2 x i64>* [[__V1_ADDR_I_I]], align 16
// CHECK-NEXT:    store <2 x i64> [[TMP43]], <2 x i64>* [[__V2_ADDR_I_I]], align 16
// CHECK-NEXT:    [[TMP44:%.*]] = load <2 x i64>, <2 x i64>* [[__V1_ADDR_I_I]], align 16
// CHECK-NEXT:    [[TMP45:%.*]] = bitcast <2 x i64> [[TMP44]] to <4 x i32>
// CHECK-NEXT:    [[TMP46:%.*]] = load <2 x i64>, <2 x i64>* [[__V2_ADDR_I_I]], align 16
// CHECK-NEXT:    [[TMP47:%.*]] = bitcast <2 x i64> [[TMP46]] to <4 x i32>
// CHECK-NEXT:    [[TMP48:%.*]] = icmp slt <4 x i32> [[TMP45]], [[TMP47]]
// CHECK-NEXT:    [[TMP49:%.*]] = select <4 x i1> [[TMP48]], <4 x i32> [[TMP45]], <4 x i32> [[TMP47]]
// CHECK-NEXT:    [[TMP50:%.*]] = bitcast <4 x i32> [[TMP49]] to <2 x i64>
// CHECK-NEXT:    store <4 x i32> [[TMP49]], <4 x i32>* [[__T10_I]], align 16
// CHECK-NEXT:    [[TMP51:%.*]] = load <4 x i32>, <4 x i32>* [[__T10_I]], align 16
// CHECK-NEXT:    [[VECEXT_I:%.*]] = extractelement <4 x i32> [[TMP51]], i32 0
// CHECK-NEXT:    ret i32 [[VECEXT_I]]
int test_mm512_reduce_min_epi32(__m512i __W){
  return _mm512_reduce_min_epi32(__W);
}

// CHECK-LABEL: define i32 @test_mm512_reduce_min_epu32(<8 x i64> %__W) #0 {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[__A_ADDR_I_I:%.*]] = alloca <4 x i64>, align 32
// CHECK-NEXT:    [[__B_ADDR_I_I:%.*]] = alloca <4 x i64>, align 32
// CHECK-NEXT:    [[__V1_ADDR_I12_I:%.*]] = alloca <2 x i64>, align 16
// CHECK-NEXT:    [[__V2_ADDR_I13_I:%.*]] = alloca <2 x i64>, align 16
// CHECK-NEXT:    [[__V1_ADDR_I10_I:%.*]] = alloca <2 x i64>, align 16
// CHECK-NEXT:    [[__V2_ADDR_I11_I:%.*]] = alloca <2 x i64>, align 16
// CHECK-NEXT:    [[__V1_ADDR_I_I:%.*]] = alloca <2 x i64>, align 16
// CHECK-NEXT:    [[__V2_ADDR_I_I:%.*]] = alloca <2 x i64>, align 16
// CHECK-NEXT:    [[__V_ADDR_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__T1_I:%.*]] = alloca <4 x i64>, align 32
// CHECK-NEXT:    [[__T2_I:%.*]] = alloca <4 x i64>, align 32
// CHECK-NEXT:    [[__T3_I:%.*]] = alloca <4 x i64>, align 32
// CHECK-NEXT:    [[__T4_I:%.*]] = alloca <2 x i64>, align 16
// CHECK-NEXT:    [[__T5_I:%.*]] = alloca <2 x i64>, align 16
// CHECK-NEXT:    [[__T6_I:%.*]] = alloca <2 x i64>, align 16
// CHECK-NEXT:    [[__T7_I:%.*]] = alloca <2 x i64>, align 16
// CHECK-NEXT:    [[__T8_I:%.*]] = alloca <2 x i64>, align 16
// CHECK-NEXT:    [[__T9_I:%.*]] = alloca <2 x i64>, align 16
// CHECK-NEXT:    [[__T10_I:%.*]] = alloca <4 x i32>, align 16
// CHECK-NEXT:    [[__W_ADDR:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    store <8 x i64> [[__W:%.*]], <8 x i64>* [[__W_ADDR]], align 64
// CHECK-NEXT:    [[TMP0:%.*]] = load <8 x i64>, <8 x i64>* [[__W_ADDR]], align 64
// CHECK-NEXT:    store <8 x i64> [[TMP0]], <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK-NEXT:    [[TMP1:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK-NEXT:    [[EXTRACT_I:%.*]] = shufflevector <8 x i64> [[TMP1]], <8 x i64> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// CHECK-NEXT:    store <4 x i64> [[EXTRACT_I]], <4 x i64>* [[__T1_I]], align 32
// CHECK-NEXT:    [[TMP2:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK-NEXT:    [[EXTRACT2_I:%.*]] = shufflevector <8 x i64> [[TMP2]], <8 x i64> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK-NEXT:    store <4 x i64> [[EXTRACT2_I]], <4 x i64>* [[__T2_I]], align 32
// CHECK-NEXT:    [[TMP3:%.*]] = load <4 x i64>, <4 x i64>* [[__T1_I]], align 32
// CHECK-NEXT:    [[TMP4:%.*]] = load <4 x i64>, <4 x i64>* [[__T2_I]], align 32
// CHECK-NEXT:    store <4 x i64> [[TMP3]], <4 x i64>* [[__A_ADDR_I_I]], align 32
// CHECK-NEXT:    store <4 x i64> [[TMP4]], <4 x i64>* [[__B_ADDR_I_I]], align 32
// CHECK-NEXT:    [[TMP5:%.*]] = load <4 x i64>, <4 x i64>* [[__A_ADDR_I_I]], align 32
// CHECK-NEXT:    [[TMP6:%.*]] = bitcast <4 x i64> [[TMP5]] to <8 x i32>
// CHECK-NEXT:    [[TMP7:%.*]] = load <4 x i64>, <4 x i64>* [[__B_ADDR_I_I]], align 32
// CHECK-NEXT:    [[TMP8:%.*]] = bitcast <4 x i64> [[TMP7]] to <8 x i32>
// CHECK-NEXT:    [[TMP9:%.*]] = icmp ult <8 x i32> [[TMP6]], [[TMP8]]
// CHECK-NEXT:    [[TMP10:%.*]] = select <8 x i1> [[TMP9]], <8 x i32> [[TMP6]], <8 x i32> [[TMP8]]
// CHECK-NEXT:    [[TMP11:%.*]] = bitcast <8 x i32> [[TMP10]] to <4 x i64>
// CHECK-NEXT:    store <4 x i64> [[TMP11]], <4 x i64>* [[__T3_I]], align 32
// CHECK-NEXT:    [[TMP12:%.*]] = load <4 x i64>, <4 x i64>* [[__T3_I]], align 32
// CHECK-NEXT:    [[EXTRACT4_I:%.*]] = shufflevector <4 x i64> [[TMP12]], <4 x i64> undef, <2 x i32> <i32 0, i32 1>
// CHECK-NEXT:    store <2 x i64> [[EXTRACT4_I]], <2 x i64>* [[__T4_I]], align 16
// CHECK-NEXT:    [[TMP13:%.*]] = load <4 x i64>, <4 x i64>* [[__T3_I]], align 32
// CHECK-NEXT:    [[EXTRACT5_I:%.*]] = shufflevector <4 x i64> [[TMP13]], <4 x i64> undef, <2 x i32> <i32 2, i32 3>
// CHECK-NEXT:    store <2 x i64> [[EXTRACT5_I]], <2 x i64>* [[__T5_I]], align 16
// CHECK-NEXT:    [[TMP14:%.*]] = load <2 x i64>, <2 x i64>* [[__T4_I]], align 16
// CHECK-NEXT:    [[TMP15:%.*]] = load <2 x i64>, <2 x i64>* [[__T5_I]], align 16
// CHECK-NEXT:    store <2 x i64> [[TMP14]], <2 x i64>* [[__V1_ADDR_I12_I]], align 16
// CHECK-NEXT:    store <2 x i64> [[TMP15]], <2 x i64>* [[__V2_ADDR_I13_I]], align 16
// CHECK-NEXT:    [[TMP16:%.*]] = load <2 x i64>, <2 x i64>* [[__V1_ADDR_I12_I]], align 16
// CHECK-NEXT:    [[TMP17:%.*]] = bitcast <2 x i64> [[TMP16]] to <4 x i32>
// CHECK-NEXT:    [[TMP18:%.*]] = load <2 x i64>, <2 x i64>* [[__V2_ADDR_I13_I]], align 16
// CHECK-NEXT:    [[TMP19:%.*]] = bitcast <2 x i64> [[TMP18]] to <4 x i32>
// CHECK-NEXT:    [[TMP20:%.*]] = icmp ult <4 x i32> [[TMP17]], [[TMP19]]
// CHECK-NEXT:    [[TMP21:%.*]] = select <4 x i1> [[TMP20]], <4 x i32> [[TMP17]], <4 x i32> [[TMP19]]
// CHECK-NEXT:    [[TMP22:%.*]] = bitcast <4 x i32> [[TMP21]] to <2 x i64>
// CHECK-NEXT:    store <2 x i64> [[TMP22]], <2 x i64>* [[__T6_I]], align 16
// CHECK-NEXT:    [[TMP23:%.*]] = load <2 x i64>, <2 x i64>* [[__T6_I]], align 16
// CHECK-NEXT:    [[TMP24:%.*]] = bitcast <2 x i64> [[TMP23]] to <4 x i32>
// CHECK-NEXT:    [[TMP25:%.*]] = load <2 x i64>, <2 x i64>* [[__T6_I]], align 16
// CHECK-NEXT:    [[TMP26:%.*]] = bitcast <2 x i64> [[TMP25]] to <4 x i32>
// CHECK-NEXT:    [[SHUFFLE_I:%.*]] = shufflevector <4 x i32> [[TMP24]], <4 x i32> [[TMP26]], <4 x i32> <i32 2, i32 3, i32 0, i32 1>
// CHECK-NEXT:    [[TMP27:%.*]] = bitcast <4 x i32> [[SHUFFLE_I]] to <2 x i64>
// CHECK-NEXT:    store <2 x i64> [[TMP27]], <2 x i64>* [[__T7_I]], align 16
// CHECK-NEXT:    [[TMP28:%.*]] = load <2 x i64>, <2 x i64>* [[__T6_I]], align 16
// CHECK-NEXT:    [[TMP29:%.*]] = load <2 x i64>, <2 x i64>* [[__T7_I]], align 16
// CHECK-NEXT:    store <2 x i64> [[TMP28]], <2 x i64>* [[__V1_ADDR_I10_I]], align 16
// CHECK-NEXT:    store <2 x i64> [[TMP29]], <2 x i64>* [[__V2_ADDR_I11_I]], align 16
// CHECK-NEXT:    [[TMP30:%.*]] = load <2 x i64>, <2 x i64>* [[__V1_ADDR_I10_I]], align 16
// CHECK-NEXT:    [[TMP31:%.*]] = bitcast <2 x i64> [[TMP30]] to <4 x i32>
// CHECK-NEXT:    [[TMP32:%.*]] = load <2 x i64>, <2 x i64>* [[__V2_ADDR_I11_I]], align 16
// CHECK-NEXT:    [[TMP33:%.*]] = bitcast <2 x i64> [[TMP32]] to <4 x i32>
// CHECK-NEXT:    [[TMP34:%.*]] = icmp ult <4 x i32> [[TMP31]], [[TMP33]]
// CHECK-NEXT:    [[TMP35:%.*]] = select <4 x i1> [[TMP34]], <4 x i32> [[TMP31]], <4 x i32> [[TMP33]]
// CHECK-NEXT:    [[TMP36:%.*]] = bitcast <4 x i32> [[TMP35]] to <2 x i64>
// CHECK-NEXT:    store <2 x i64> [[TMP36]], <2 x i64>* [[__T8_I]], align 16
// CHECK-NEXT:    [[TMP37:%.*]] = load <2 x i64>, <2 x i64>* [[__T8_I]], align 16
// CHECK-NEXT:    [[TMP38:%.*]] = bitcast <2 x i64> [[TMP37]] to <4 x i32>
// CHECK-NEXT:    [[TMP39:%.*]] = load <2 x i64>, <2 x i64>* [[__T8_I]], align 16
// CHECK-NEXT:    [[TMP40:%.*]] = bitcast <2 x i64> [[TMP39]] to <4 x i32>
// CHECK-NEXT:    [[SHUFFLE8_I:%.*]] = shufflevector <4 x i32> [[TMP38]], <4 x i32> [[TMP40]], <4 x i32> <i32 1, i32 0, i32 3, i32 2>
// CHECK-NEXT:    [[TMP41:%.*]] = bitcast <4 x i32> [[SHUFFLE8_I]] to <2 x i64>
// CHECK-NEXT:    store <2 x i64> [[TMP41]], <2 x i64>* [[__T9_I]], align 16
// CHECK-NEXT:    [[TMP42:%.*]] = load <2 x i64>, <2 x i64>* [[__T8_I]], align 16
// CHECK-NEXT:    [[TMP43:%.*]] = load <2 x i64>, <2 x i64>* [[__T9_I]], align 16
// CHECK-NEXT:    store <2 x i64> [[TMP42]], <2 x i64>* [[__V1_ADDR_I_I]], align 16
// CHECK-NEXT:    store <2 x i64> [[TMP43]], <2 x i64>* [[__V2_ADDR_I_I]], align 16
// CHECK-NEXT:    [[TMP44:%.*]] = load <2 x i64>, <2 x i64>* [[__V1_ADDR_I_I]], align 16
// CHECK-NEXT:    [[TMP45:%.*]] = bitcast <2 x i64> [[TMP44]] to <4 x i32>
// CHECK-NEXT:    [[TMP46:%.*]] = load <2 x i64>, <2 x i64>* [[__V2_ADDR_I_I]], align 16
// CHECK-NEXT:    [[TMP47:%.*]] = bitcast <2 x i64> [[TMP46]] to <4 x i32>
// CHECK-NEXT:    [[TMP48:%.*]] = icmp ult <4 x i32> [[TMP45]], [[TMP47]]
// CHECK-NEXT:    [[TMP49:%.*]] = select <4 x i1> [[TMP48]], <4 x i32> [[TMP45]], <4 x i32> [[TMP47]]
// CHECK-NEXT:    [[TMP50:%.*]] = bitcast <4 x i32> [[TMP49]] to <2 x i64>
// CHECK-NEXT:    store <4 x i32> [[TMP49]], <4 x i32>* [[__T10_I]], align 16
// CHECK-NEXT:    [[TMP51:%.*]] = load <4 x i32>, <4 x i32>* [[__T10_I]], align 16
// CHECK-NEXT:    [[VECEXT_I:%.*]] = extractelement <4 x i32> [[TMP51]], i32 0
// CHECK-NEXT:    ret i32 [[VECEXT_I]]
unsigned int test_mm512_reduce_min_epu32(__m512i __W){
  return _mm512_reduce_min_epu32(__W); 
}

// CHECK-LABEL: define float @test_mm512_reduce_min_ps(<16 x float> %__W) #0 {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[__A_ADDR_I14_I:%.*]] = alloca <8 x float>, align 32
// CHECK-NEXT:    [[__B_ADDR_I15_I:%.*]] = alloca <8 x float>, align 32
// CHECK-NEXT:    [[__A_ADDR_I12_I:%.*]] = alloca <4 x float>, align 16
// CHECK-NEXT:    [[__B_ADDR_I13_I:%.*]] = alloca <4 x float>, align 16
// CHECK-NEXT:    [[__A_ADDR_I10_I:%.*]] = alloca <4 x float>, align 16
// CHECK-NEXT:    [[__B_ADDR_I11_I:%.*]] = alloca <4 x float>, align 16
// CHECK-NEXT:    [[__A_ADDR_I_I:%.*]] = alloca <4 x float>, align 16
// CHECK-NEXT:    [[__B_ADDR_I_I:%.*]] = alloca <4 x float>, align 16
// CHECK-NEXT:    [[__V_ADDR_I:%.*]] = alloca <16 x float>, align 64
// CHECK-NEXT:    [[__T1_I:%.*]] = alloca <8 x float>, align 32
// CHECK-NEXT:    [[__T2_I:%.*]] = alloca <8 x float>, align 32
// CHECK-NEXT:    [[__T3_I:%.*]] = alloca <8 x float>, align 32
// CHECK-NEXT:    [[__T4_I:%.*]] = alloca <4 x float>, align 16
// CHECK-NEXT:    [[__T5_I:%.*]] = alloca <4 x float>, align 16
// CHECK-NEXT:    [[__T6_I:%.*]] = alloca <4 x float>, align 16
// CHECK-NEXT:    [[__T7_I:%.*]] = alloca <4 x float>, align 16
// CHECK-NEXT:    [[__T8_I:%.*]] = alloca <4 x float>, align 16
// CHECK-NEXT:    [[__T9_I:%.*]] = alloca <4 x float>, align 16
// CHECK-NEXT:    [[__T10_I:%.*]] = alloca <4 x float>, align 16
// CHECK-NEXT:    [[__W_ADDR:%.*]] = alloca <16 x float>, align 64
// CHECK-NEXT:    store <16 x float> [[__W:%.*]], <16 x float>* [[__W_ADDR]], align 64
// CHECK-NEXT:    [[TMP0:%.*]] = load <16 x float>, <16 x float>* [[__W_ADDR]], align 64
// CHECK-NEXT:    store <16 x float> [[TMP0]], <16 x float>* [[__V_ADDR_I]], align 64
// CHECK-NEXT:    [[TMP1:%.*]] = load <16 x float>, <16 x float>* [[__V_ADDR_I]], align 64
// CHECK-NEXT:    [[TMP2:%.*]] = bitcast <16 x float> [[TMP1]] to <8 x double>
// CHECK-NEXT:    [[EXTRACT_I:%.*]] = shufflevector <8 x double> [[TMP2]], <8 x double> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// CHECK-NEXT:    [[TMP3:%.*]] = bitcast <4 x double> [[EXTRACT_I]] to <8 x float>
// CHECK-NEXT:    store <8 x float> [[TMP3]], <8 x float>* [[__T1_I]], align 32
// CHECK-NEXT:    [[TMP4:%.*]] = load <16 x float>, <16 x float>* [[__V_ADDR_I]], align 64
// CHECK-NEXT:    [[TMP5:%.*]] = bitcast <16 x float> [[TMP4]] to <8 x double>
// CHECK-NEXT:    [[EXTRACT2_I:%.*]] = shufflevector <8 x double> [[TMP5]], <8 x double> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK-NEXT:    [[TMP6:%.*]] = bitcast <4 x double> [[EXTRACT2_I]] to <8 x float>
// CHECK-NEXT:    store <8 x float> [[TMP6]], <8 x float>* [[__T2_I]], align 32
// CHECK-NEXT:    [[TMP7:%.*]] = load <8 x float>, <8 x float>* [[__T1_I]], align 32
// CHECK-NEXT:    [[TMP8:%.*]] = load <8 x float>, <8 x float>* [[__T2_I]], align 32
// CHECK-NEXT:    store <8 x float> [[TMP7]], <8 x float>* [[__A_ADDR_I14_I]], align 32
// CHECK-NEXT:    store <8 x float> [[TMP8]], <8 x float>* [[__B_ADDR_I15_I]], align 32
// CHECK-NEXT:    [[TMP9:%.*]] = load <8 x float>, <8 x float>* [[__A_ADDR_I14_I]], align 32
// CHECK-NEXT:    [[TMP10:%.*]] = load <8 x float>, <8 x float>* [[__B_ADDR_I15_I]], align 32
// CHECK-NEXT:    [[TMP11:%.*]] = call <8 x float> @llvm.x86.avx.min.ps.256(<8 x float> [[TMP9]], <8 x float> [[TMP10]]) #2
// CHECK-NEXT:    store <8 x float> [[TMP11]], <8 x float>* [[__T3_I]], align 32
// CHECK-NEXT:    [[TMP12:%.*]] = load <8 x float>, <8 x float>* [[__T3_I]], align 32
// CHECK-NEXT:    [[EXTRACT4_I:%.*]] = shufflevector <8 x float> [[TMP12]], <8 x float> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// CHECK-NEXT:    store <4 x float> [[EXTRACT4_I]], <4 x float>* [[__T4_I]], align 16
// CHECK-NEXT:    [[TMP13:%.*]] = load <8 x float>, <8 x float>* [[__T3_I]], align 32
// CHECK-NEXT:    [[EXTRACT5_I:%.*]] = shufflevector <8 x float> [[TMP13]], <8 x float> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK-NEXT:    store <4 x float> [[EXTRACT5_I]], <4 x float>* [[__T5_I]], align 16
// CHECK-NEXT:    [[TMP14:%.*]] = load <4 x float>, <4 x float>* [[__T4_I]], align 16
// CHECK-NEXT:    [[TMP15:%.*]] = load <4 x float>, <4 x float>* [[__T5_I]], align 16
// CHECK-NEXT:    store <4 x float> [[TMP14]], <4 x float>* [[__A_ADDR_I12_I]], align 16
// CHECK-NEXT:    store <4 x float> [[TMP15]], <4 x float>* [[__B_ADDR_I13_I]], align 16
// CHECK-NEXT:    [[TMP16:%.*]] = load <4 x float>, <4 x float>* [[__A_ADDR_I12_I]], align 16
// CHECK-NEXT:    [[TMP17:%.*]] = load <4 x float>, <4 x float>* [[__B_ADDR_I13_I]], align 16
// CHECK-NEXT:    [[TMP18:%.*]] = call <4 x float> @llvm.x86.sse.min.ps(<4 x float> [[TMP16]], <4 x float> [[TMP17]]) #2
// CHECK-NEXT:    store <4 x float> [[TMP18]], <4 x float>* [[__T6_I]], align 16
// CHECK-NEXT:    [[TMP19:%.*]] = load <4 x float>, <4 x float>* [[__T6_I]], align 16
// CHECK-NEXT:    [[TMP20:%.*]] = load <4 x float>, <4 x float>* [[__T6_I]], align 16
// CHECK-NEXT:    [[SHUFFLE_I:%.*]] = shufflevector <4 x float> [[TMP19]], <4 x float> [[TMP20]], <4 x i32> <i32 2, i32 3, i32 0, i32 1>
// CHECK-NEXT:    store <4 x float> [[SHUFFLE_I]], <4 x float>* [[__T7_I]], align 16
// CHECK-NEXT:    [[TMP21:%.*]] = load <4 x float>, <4 x float>* [[__T6_I]], align 16
// CHECK-NEXT:    [[TMP22:%.*]] = load <4 x float>, <4 x float>* [[__T7_I]], align 16
// CHECK-NEXT:    store <4 x float> [[TMP21]], <4 x float>* [[__A_ADDR_I10_I]], align 16
// CHECK-NEXT:    store <4 x float> [[TMP22]], <4 x float>* [[__B_ADDR_I11_I]], align 16
// CHECK-NEXT:    [[TMP23:%.*]] = load <4 x float>, <4 x float>* [[__A_ADDR_I10_I]], align 16
// CHECK-NEXT:    [[TMP24:%.*]] = load <4 x float>, <4 x float>* [[__B_ADDR_I11_I]], align 16
// CHECK-NEXT:    [[TMP25:%.*]] = call <4 x float> @llvm.x86.sse.min.ps(<4 x float> [[TMP23]], <4 x float> [[TMP24]]) #2
// CHECK-NEXT:    store <4 x float> [[TMP25]], <4 x float>* [[__T8_I]], align 16
// CHECK-NEXT:    [[TMP26:%.*]] = load <4 x float>, <4 x float>* [[__T8_I]], align 16
// CHECK-NEXT:    [[TMP27:%.*]] = load <4 x float>, <4 x float>* [[__T8_I]], align 16
// CHECK-NEXT:    [[SHUFFLE8_I:%.*]] = shufflevector <4 x float> [[TMP26]], <4 x float> [[TMP27]], <4 x i32> <i32 1, i32 0, i32 3, i32 2>
// CHECK-NEXT:    store <4 x float> [[SHUFFLE8_I]], <4 x float>* [[__T9_I]], align 16
// CHECK-NEXT:    [[TMP28:%.*]] = load <4 x float>, <4 x float>* [[__T8_I]], align 16
// CHECK-NEXT:    [[TMP29:%.*]] = load <4 x float>, <4 x float>* [[__T9_I]], align 16
// CHECK-NEXT:    store <4 x float> [[TMP28]], <4 x float>* [[__A_ADDR_I_I]], align 16
// CHECK-NEXT:    store <4 x float> [[TMP29]], <4 x float>* [[__B_ADDR_I_I]], align 16
// CHECK-NEXT:    [[TMP30:%.*]] = load <4 x float>, <4 x float>* [[__A_ADDR_I_I]], align 16
// CHECK-NEXT:    [[TMP31:%.*]] = load <4 x float>, <4 x float>* [[__B_ADDR_I_I]], align 16
// CHECK-NEXT:    [[TMP32:%.*]] = call <4 x float> @llvm.x86.sse.min.ps(<4 x float> [[TMP30]], <4 x float> [[TMP31]]) #2
// CHECK-NEXT:    store <4 x float> [[TMP32]], <4 x float>* [[__T10_I]], align 16
// CHECK-NEXT:    [[TMP33:%.*]] = load <4 x float>, <4 x float>* [[__T10_I]], align 16
// CHECK-NEXT:    [[VECEXT_I:%.*]] = extractelement <4 x float> [[TMP33]], i32 0
// CHECK-NEXT:    ret float [[VECEXT_I]]
float test_mm512_reduce_min_ps(__m512 __W){
  return _mm512_reduce_min_ps(__W); 
}

// CHECK-LABEL: define i32 @test_mm512_mask_reduce_max_epi32(i16 zeroext %__M, <8 x i64> %__W) #0 {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[__W_ADDR_I_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__U_ADDR_I_I:%.*]] = alloca i16, align 2
// CHECK-NEXT:    [[__A2_ADDR_I_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__A_ADDR_I_I:%.*]] = alloca <4 x i64>, align 32
// CHECK-NEXT:    [[__B_ADDR_I_I:%.*]] = alloca <4 x i64>, align 32
// CHECK-NEXT:    [[__V1_ADDR_I14_I:%.*]] = alloca <2 x i64>, align 16
// CHECK-NEXT:    [[__V2_ADDR_I15_I:%.*]] = alloca <2 x i64>, align 16
// CHECK-NEXT:    [[__V1_ADDR_I12_I:%.*]] = alloca <2 x i64>, align 16
// CHECK-NEXT:    [[__V2_ADDR_I13_I:%.*]] = alloca <2 x i64>, align 16
// CHECK-NEXT:    [[__V1_ADDR_I_I:%.*]] = alloca <2 x i64>, align 16
// CHECK-NEXT:    [[__V2_ADDR_I_I:%.*]] = alloca <2 x i64>, align 16
// CHECK-NEXT:    [[__S_ADDR_I_I:%.*]] = alloca i32, align 4
// CHECK-NEXT:    [[DOTCOMPOUNDLITERAL_I_I:%.*]] = alloca <16 x i32>, align 64
// CHECK-NEXT:    [[__M_ADDR_I:%.*]] = alloca i16, align 2
// CHECK-NEXT:    [[__V_ADDR_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__T1_I:%.*]] = alloca <4 x i64>, align 32
// CHECK-NEXT:    [[__T2_I:%.*]] = alloca <4 x i64>, align 32
// CHECK-NEXT:    [[__T3_I:%.*]] = alloca <4 x i64>, align 32
// CHECK-NEXT:    [[__T4_I:%.*]] = alloca <2 x i64>, align 16
// CHECK-NEXT:    [[__T5_I:%.*]] = alloca <2 x i64>, align 16
// CHECK-NEXT:    [[__T6_I:%.*]] = alloca <2 x i64>, align 16
// CHECK-NEXT:    [[__T7_I:%.*]] = alloca <2 x i64>, align 16
// CHECK-NEXT:    [[__T8_I:%.*]] = alloca <2 x i64>, align 16
// CHECK-NEXT:    [[__T9_I:%.*]] = alloca <2 x i64>, align 16
// CHECK-NEXT:    [[__T10_I:%.*]] = alloca <4 x i32>, align 16
// CHECK-NEXT:    [[__M_ADDR:%.*]] = alloca i16, align 2
// CHECK-NEXT:    [[__W_ADDR:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    store i16 [[__M:%.*]], i16* [[__M_ADDR]], align 2
// CHECK-NEXT:    store <8 x i64> [[__W:%.*]], <8 x i64>* [[__W_ADDR]], align 64
// CHECK-NEXT:    [[TMP0:%.*]] = load i16, i16* [[__M_ADDR]], align 2
// CHECK-NEXT:    [[TMP1:%.*]] = load <8 x i64>, <8 x i64>* [[__W_ADDR]], align 64
// CHECK-NEXT:    store i16 [[TMP0]], i16* [[__M_ADDR_I]], align 2
// CHECK-NEXT:    store <8 x i64> [[TMP1]], <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK-NEXT:    store i32 -2147483648, i32* [[__S_ADDR_I_I]], align 4
// CHECK-NEXT:    [[TMP2:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK-NEXT:    [[VECINIT_I_I:%.*]] = insertelement <16 x i32> undef, i32 [[TMP2]], i32 0
// CHECK-NEXT:    [[TMP3:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK-NEXT:    [[VECINIT1_I_I:%.*]] = insertelement <16 x i32> [[VECINIT_I_I]], i32 [[TMP3]], i32 1
// CHECK-NEXT:    [[TMP4:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK-NEXT:    [[VECINIT2_I_I:%.*]] = insertelement <16 x i32> [[VECINIT1_I_I]], i32 [[TMP4]], i32 2
// CHECK-NEXT:    [[TMP5:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK-NEXT:    [[VECINIT3_I_I:%.*]] = insertelement <16 x i32> [[VECINIT2_I_I]], i32 [[TMP5]], i32 3
// CHECK-NEXT:    [[TMP6:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK-NEXT:    [[VECINIT4_I_I:%.*]] = insertelement <16 x i32> [[VECINIT3_I_I]], i32 [[TMP6]], i32 4
// CHECK-NEXT:    [[TMP7:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK-NEXT:    [[VECINIT5_I_I:%.*]] = insertelement <16 x i32> [[VECINIT4_I_I]], i32 [[TMP7]], i32 5
// CHECK-NEXT:    [[TMP8:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK-NEXT:    [[VECINIT6_I_I:%.*]] = insertelement <16 x i32> [[VECINIT5_I_I]], i32 [[TMP8]], i32 6
// CHECK-NEXT:    [[TMP9:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK-NEXT:    [[VECINIT7_I_I:%.*]] = insertelement <16 x i32> [[VECINIT6_I_I]], i32 [[TMP9]], i32 7
// CHECK-NEXT:    [[TMP10:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK-NEXT:    [[VECINIT8_I_I:%.*]] = insertelement <16 x i32> [[VECINIT7_I_I]], i32 [[TMP10]], i32 8
// CHECK-NEXT:    [[TMP11:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK-NEXT:    [[VECINIT9_I_I:%.*]] = insertelement <16 x i32> [[VECINIT8_I_I]], i32 [[TMP11]], i32 9
// CHECK-NEXT:    [[TMP12:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK-NEXT:    [[VECINIT10_I_I:%.*]] = insertelement <16 x i32> [[VECINIT9_I_I]], i32 [[TMP12]], i32 10
// CHECK-NEXT:    [[TMP13:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK-NEXT:    [[VECINIT11_I_I:%.*]] = insertelement <16 x i32> [[VECINIT10_I_I]], i32 [[TMP13]], i32 11
// CHECK-NEXT:    [[TMP14:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK-NEXT:    [[VECINIT12_I_I:%.*]] = insertelement <16 x i32> [[VECINIT11_I_I]], i32 [[TMP14]], i32 12
// CHECK-NEXT:    [[TMP15:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK-NEXT:    [[VECINIT13_I_I:%.*]] = insertelement <16 x i32> [[VECINIT12_I_I]], i32 [[TMP15]], i32 13
// CHECK-NEXT:    [[TMP16:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK-NEXT:    [[VECINIT14_I_I:%.*]] = insertelement <16 x i32> [[VECINIT13_I_I]], i32 [[TMP16]], i32 14
// CHECK-NEXT:    [[TMP17:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK-NEXT:    [[VECINIT15_I_I:%.*]] = insertelement <16 x i32> [[VECINIT14_I_I]], i32 [[TMP17]], i32 15
// CHECK-NEXT:    store <16 x i32> [[VECINIT15_I_I]], <16 x i32>* [[DOTCOMPOUNDLITERAL_I_I]], align 64
// CHECK-NEXT:    [[TMP18:%.*]] = load <16 x i32>, <16 x i32>* [[DOTCOMPOUNDLITERAL_I_I]], align 64
// CHECK-NEXT:    [[TMP19:%.*]] = bitcast <16 x i32> [[TMP18]] to <8 x i64>
// CHECK-NEXT:    [[TMP20:%.*]] = load i16, i16* [[__M_ADDR_I]], align 2
// CHECK-NEXT:    [[TMP21:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK-NEXT:    store <8 x i64> [[TMP19]], <8 x i64>* [[__W_ADDR_I_I]], align 64
// CHECK-NEXT:    store i16 [[TMP20]], i16* [[__U_ADDR_I_I]], align 2
// CHECK-NEXT:    store <8 x i64> [[TMP21]], <8 x i64>* [[__A2_ADDR_I_I]], align 64
// CHECK-NEXT:    [[TMP22:%.*]] = load i16, i16* [[__U_ADDR_I_I]], align 2
// CHECK-NEXT:    [[TMP23:%.*]] = load <8 x i64>, <8 x i64>* [[__A2_ADDR_I_I]], align 64
// CHECK-NEXT:    [[TMP24:%.*]] = bitcast <8 x i64> [[TMP23]] to <16 x i32>
// CHECK-NEXT:    [[TMP25:%.*]] = load <8 x i64>, <8 x i64>* [[__W_ADDR_I_I]], align 64
// CHECK-NEXT:    [[TMP26:%.*]] = bitcast <8 x i64> [[TMP25]] to <16 x i32>
// CHECK-NEXT:    [[TMP27:%.*]] = bitcast i16 [[TMP22]] to <16 x i1>
// CHECK-NEXT:    [[TMP28:%.*]] = select <16 x i1> [[TMP27]], <16 x i32> [[TMP24]], <16 x i32> [[TMP26]]
// CHECK-NEXT:    [[TMP29:%.*]] = bitcast <16 x i32> [[TMP28]] to <8 x i64>
// CHECK-NEXT:    store <8 x i64> [[TMP29]], <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK-NEXT:    [[TMP30:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK-NEXT:    [[EXTRACT_I:%.*]] = shufflevector <8 x i64> [[TMP30]], <8 x i64> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// CHECK-NEXT:    store <4 x i64> [[EXTRACT_I]], <4 x i64>* [[__T1_I]], align 32
// CHECK-NEXT:    [[TMP31:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK-NEXT:    [[EXTRACT4_I:%.*]] = shufflevector <8 x i64> [[TMP31]], <8 x i64> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK-NEXT:    store <4 x i64> [[EXTRACT4_I]], <4 x i64>* [[__T2_I]], align 32
// CHECK-NEXT:    [[TMP32:%.*]] = load <4 x i64>, <4 x i64>* [[__T1_I]], align 32
// CHECK-NEXT:    [[TMP33:%.*]] = load <4 x i64>, <4 x i64>* [[__T2_I]], align 32
// CHECK-NEXT:    store <4 x i64> [[TMP32]], <4 x i64>* [[__A_ADDR_I_I]], align 32
// CHECK-NEXT:    store <4 x i64> [[TMP33]], <4 x i64>* [[__B_ADDR_I_I]], align 32
// CHECK-NEXT:    [[TMP34:%.*]] = load <4 x i64>, <4 x i64>* [[__A_ADDR_I_I]], align 32
// CHECK-NEXT:    [[TMP35:%.*]] = bitcast <4 x i64> [[TMP34]] to <8 x i32>
// CHECK-NEXT:    [[TMP36:%.*]] = load <4 x i64>, <4 x i64>* [[__B_ADDR_I_I]], align 32
// CHECK-NEXT:    [[TMP37:%.*]] = bitcast <4 x i64> [[TMP36]] to <8 x i32>
// CHECK-NEXT:    [[TMP38:%.*]] = icmp sgt <8 x i32> [[TMP35]], [[TMP37]]
// CHECK-NEXT:    [[TMP39:%.*]] = select <8 x i1> [[TMP38]], <8 x i32> [[TMP35]], <8 x i32> [[TMP37]]
// CHECK-NEXT:    [[TMP40:%.*]] = bitcast <8 x i32> [[TMP39]] to <4 x i64>
// CHECK-NEXT:    store <4 x i64> [[TMP40]], <4 x i64>* [[__T3_I]], align 32
// CHECK-NEXT:    [[TMP41:%.*]] = load <4 x i64>, <4 x i64>* [[__T3_I]], align 32
// CHECK-NEXT:    [[EXTRACT6_I:%.*]] = shufflevector <4 x i64> [[TMP41]], <4 x i64> undef, <2 x i32> <i32 0, i32 1>
// CHECK-NEXT:    store <2 x i64> [[EXTRACT6_I]], <2 x i64>* [[__T4_I]], align 16
// CHECK-NEXT:    [[TMP42:%.*]] = load <4 x i64>, <4 x i64>* [[__T3_I]], align 32
// CHECK-NEXT:    [[EXTRACT7_I:%.*]] = shufflevector <4 x i64> [[TMP42]], <4 x i64> undef, <2 x i32> <i32 2, i32 3>
// CHECK-NEXT:    store <2 x i64> [[EXTRACT7_I]], <2 x i64>* [[__T5_I]], align 16
// CHECK-NEXT:    [[TMP43:%.*]] = load <2 x i64>, <2 x i64>* [[__T4_I]], align 16
// CHECK-NEXT:    [[TMP44:%.*]] = load <2 x i64>, <2 x i64>* [[__T5_I]], align 16
// CHECK-NEXT:    store <2 x i64> [[TMP43]], <2 x i64>* [[__V1_ADDR_I14_I]], align 16
// CHECK-NEXT:    store <2 x i64> [[TMP44]], <2 x i64>* [[__V2_ADDR_I15_I]], align 16
// CHECK-NEXT:    [[TMP45:%.*]] = load <2 x i64>, <2 x i64>* [[__V1_ADDR_I14_I]], align 16
// CHECK-NEXT:    [[TMP46:%.*]] = bitcast <2 x i64> [[TMP45]] to <4 x i32>
// CHECK-NEXT:    [[TMP47:%.*]] = load <2 x i64>, <2 x i64>* [[__V2_ADDR_I15_I]], align 16
// CHECK-NEXT:    [[TMP48:%.*]] = bitcast <2 x i64> [[TMP47]] to <4 x i32>
// CHECK-NEXT:    [[TMP49:%.*]] = icmp sgt <4 x i32> [[TMP46]], [[TMP48]]
// CHECK-NEXT:    [[TMP50:%.*]] = select <4 x i1> [[TMP49]], <4 x i32> [[TMP46]], <4 x i32> [[TMP48]]
// CHECK-NEXT:    [[TMP51:%.*]] = bitcast <4 x i32> [[TMP50]] to <2 x i64>
// CHECK-NEXT:    store <2 x i64> [[TMP51]], <2 x i64>* [[__T6_I]], align 16
// CHECK-NEXT:    [[TMP52:%.*]] = load <2 x i64>, <2 x i64>* [[__T6_I]], align 16
// CHECK-NEXT:    [[TMP53:%.*]] = bitcast <2 x i64> [[TMP52]] to <4 x i32>
// CHECK-NEXT:    [[TMP54:%.*]] = load <2 x i64>, <2 x i64>* [[__T6_I]], align 16
// CHECK-NEXT:    [[TMP55:%.*]] = bitcast <2 x i64> [[TMP54]] to <4 x i32>
// CHECK-NEXT:    [[SHUFFLE_I:%.*]] = shufflevector <4 x i32> [[TMP53]], <4 x i32> [[TMP55]], <4 x i32> <i32 2, i32 3, i32 0, i32 1>
// CHECK-NEXT:    [[TMP56:%.*]] = bitcast <4 x i32> [[SHUFFLE_I]] to <2 x i64>
// CHECK-NEXT:    store <2 x i64> [[TMP56]], <2 x i64>* [[__T7_I]], align 16
// CHECK-NEXT:    [[TMP57:%.*]] = load <2 x i64>, <2 x i64>* [[__T6_I]], align 16
// CHECK-NEXT:    [[TMP58:%.*]] = load <2 x i64>, <2 x i64>* [[__T7_I]], align 16
// CHECK-NEXT:    store <2 x i64> [[TMP57]], <2 x i64>* [[__V1_ADDR_I12_I]], align 16
// CHECK-NEXT:    store <2 x i64> [[TMP58]], <2 x i64>* [[__V2_ADDR_I13_I]], align 16
// CHECK-NEXT:    [[TMP59:%.*]] = load <2 x i64>, <2 x i64>* [[__V1_ADDR_I12_I]], align 16
// CHECK-NEXT:    [[TMP60:%.*]] = bitcast <2 x i64> [[TMP59]] to <4 x i32>
// CHECK-NEXT:    [[TMP61:%.*]] = load <2 x i64>, <2 x i64>* [[__V2_ADDR_I13_I]], align 16
// CHECK-NEXT:    [[TMP62:%.*]] = bitcast <2 x i64> [[TMP61]] to <4 x i32>
// CHECK-NEXT:    [[TMP63:%.*]] = icmp sgt <4 x i32> [[TMP60]], [[TMP62]]
// CHECK-NEXT:    [[TMP64:%.*]] = select <4 x i1> [[TMP63]], <4 x i32> [[TMP60]], <4 x i32> [[TMP62]]
// CHECK-NEXT:    [[TMP65:%.*]] = bitcast <4 x i32> [[TMP64]] to <2 x i64>
// CHECK-NEXT:    store <2 x i64> [[TMP65]], <2 x i64>* [[__T8_I]], align 16
// CHECK-NEXT:    [[TMP66:%.*]] = load <2 x i64>, <2 x i64>* [[__T8_I]], align 16
// CHECK-NEXT:    [[TMP67:%.*]] = bitcast <2 x i64> [[TMP66]] to <4 x i32>
// CHECK-NEXT:    [[TMP68:%.*]] = load <2 x i64>, <2 x i64>* [[__T8_I]], align 16
// CHECK-NEXT:    [[TMP69:%.*]] = bitcast <2 x i64> [[TMP68]] to <4 x i32>
// CHECK-NEXT:    [[SHUFFLE10_I:%.*]] = shufflevector <4 x i32> [[TMP67]], <4 x i32> [[TMP69]], <4 x i32> <i32 1, i32 0, i32 3, i32 2>
// CHECK-NEXT:    [[TMP70:%.*]] = bitcast <4 x i32> [[SHUFFLE10_I]] to <2 x i64>
// CHECK-NEXT:    store <2 x i64> [[TMP70]], <2 x i64>* [[__T9_I]], align 16
// CHECK-NEXT:    [[TMP71:%.*]] = load <2 x i64>, <2 x i64>* [[__T8_I]], align 16
// CHECK-NEXT:    [[TMP72:%.*]] = load <2 x i64>, <2 x i64>* [[__T9_I]], align 16
// CHECK-NEXT:    store <2 x i64> [[TMP71]], <2 x i64>* [[__V1_ADDR_I_I]], align 16
// CHECK-NEXT:    store <2 x i64> [[TMP72]], <2 x i64>* [[__V2_ADDR_I_I]], align 16
// CHECK-NEXT:    [[TMP73:%.*]] = load <2 x i64>, <2 x i64>* [[__V1_ADDR_I_I]], align 16
// CHECK-NEXT:    [[TMP74:%.*]] = bitcast <2 x i64> [[TMP73]] to <4 x i32>
// CHECK-NEXT:    [[TMP75:%.*]] = load <2 x i64>, <2 x i64>* [[__V2_ADDR_I_I]], align 16
// CHECK-NEXT:    [[TMP76:%.*]] = bitcast <2 x i64> [[TMP75]] to <4 x i32>
// CHECK-NEXT:    [[TMP77:%.*]] = icmp sgt <4 x i32> [[TMP74]], [[TMP76]]
// CHECK-NEXT:    [[TMP78:%.*]] = select <4 x i1> [[TMP77]], <4 x i32> [[TMP74]], <4 x i32> [[TMP76]]
// CHECK-NEXT:    [[TMP79:%.*]] = bitcast <4 x i32> [[TMP78]] to <2 x i64>
// CHECK-NEXT:    store <4 x i32> [[TMP78]], <4 x i32>* [[__T10_I]], align 16
// CHECK-NEXT:    [[TMP80:%.*]] = load <4 x i32>, <4 x i32>* [[__T10_I]], align 16
// CHECK-NEXT:    [[VECEXT_I:%.*]] = extractelement <4 x i32> [[TMP80]], i32 0
// CHECK-NEXT:    ret i32 [[VECEXT_I]]
int test_mm512_mask_reduce_max_epi32(__mmask16 __M, __m512i __W){
  return _mm512_mask_reduce_max_epi32(__M, __W); 
}

// CHECK-LABEL: define i32 @test_mm512_mask_reduce_max_epu32(i16 zeroext %__M, <8 x i64> %__W) #0 {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[__A2_ADDR_I_I:%.*]] = alloca <4 x i64>, align 32
// CHECK-NEXT:    [[__B_ADDR_I_I:%.*]] = alloca <4 x i64>, align 32
// CHECK-NEXT:    [[__V1_ADDR_I13_I:%.*]] = alloca <2 x i64>, align 16
// CHECK-NEXT:    [[__V2_ADDR_I14_I:%.*]] = alloca <2 x i64>, align 16
// CHECK-NEXT:    [[__V1_ADDR_I11_I:%.*]] = alloca <2 x i64>, align 16
// CHECK-NEXT:    [[__V2_ADDR_I12_I:%.*]] = alloca <2 x i64>, align 16
// CHECK-NEXT:    [[__V1_ADDR_I_I:%.*]] = alloca <2 x i64>, align 16
// CHECK-NEXT:    [[__V2_ADDR_I_I:%.*]] = alloca <2 x i64>, align 16
// CHECK-NEXT:    [[DOTCOMPOUNDLITERAL_I_I_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__U_ADDR_I_I:%.*]] = alloca i16, align 2
// CHECK-NEXT:    [[__A_ADDR_I_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__M_ADDR_I:%.*]] = alloca i16, align 2
// CHECK-NEXT:    [[__V_ADDR_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__T1_I:%.*]] = alloca <4 x i64>, align 32
// CHECK-NEXT:    [[__T2_I:%.*]] = alloca <4 x i64>, align 32
// CHECK-NEXT:    [[__T3_I:%.*]] = alloca <4 x i64>, align 32
// CHECK-NEXT:    [[__T4_I:%.*]] = alloca <2 x i64>, align 16
// CHECK-NEXT:    [[__T5_I:%.*]] = alloca <2 x i64>, align 16
// CHECK-NEXT:    [[__T6_I:%.*]] = alloca <2 x i64>, align 16
// CHECK-NEXT:    [[__T7_I:%.*]] = alloca <2 x i64>, align 16
// CHECK-NEXT:    [[__T8_I:%.*]] = alloca <2 x i64>, align 16
// CHECK-NEXT:    [[__T9_I:%.*]] = alloca <2 x i64>, align 16
// CHECK-NEXT:    [[__T10_I:%.*]] = alloca <4 x i32>, align 16
// CHECK-NEXT:    [[__M_ADDR:%.*]] = alloca i16, align 2
// CHECK-NEXT:    [[__W_ADDR:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    store i16 [[__M:%.*]], i16* [[__M_ADDR]], align 2
// CHECK-NEXT:    store <8 x i64> [[__W:%.*]], <8 x i64>* [[__W_ADDR]], align 64
// CHECK-NEXT:    [[TMP0:%.*]] = load i16, i16* [[__M_ADDR]], align 2
// CHECK-NEXT:    [[TMP1:%.*]] = load <8 x i64>, <8 x i64>* [[__W_ADDR]], align 64
// CHECK-NEXT:    store i16 [[TMP0]], i16* [[__M_ADDR_I]], align 2
// CHECK-NEXT:    store <8 x i64> [[TMP1]], <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK-NEXT:    [[TMP2:%.*]] = load i16, i16* [[__M_ADDR_I]], align 2
// CHECK-NEXT:    [[TMP3:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK-NEXT:    store i16 [[TMP2]], i16* [[__U_ADDR_I_I]], align 2
// CHECK-NEXT:    store <8 x i64> [[TMP3]], <8 x i64>* [[__A_ADDR_I_I]], align 64
// CHECK-NEXT:    [[TMP4:%.*]] = load i16, i16* [[__U_ADDR_I_I]], align 2
// CHECK-NEXT:    [[TMP5:%.*]] = load <8 x i64>, <8 x i64>* [[__A_ADDR_I_I]], align 64
// CHECK-NEXT:    [[TMP6:%.*]] = bitcast <8 x i64> [[TMP5]] to <16 x i32>
// CHECK-NEXT:    store <8 x i64> zeroinitializer, <8 x i64>* [[DOTCOMPOUNDLITERAL_I_I_I]], align 64
// CHECK-NEXT:    [[TMP7:%.*]] = load <8 x i64>, <8 x i64>* [[DOTCOMPOUNDLITERAL_I_I_I]], align 64
// CHECK-NEXT:    [[TMP8:%.*]] = bitcast <8 x i64> [[TMP7]] to <16 x i32>
// CHECK-NEXT:    [[TMP9:%.*]] = bitcast i16 [[TMP4]] to <16 x i1>
// CHECK-NEXT:    [[TMP10:%.*]] = select <16 x i1> [[TMP9]], <16 x i32> [[TMP6]], <16 x i32> [[TMP8]]
// CHECK-NEXT:    [[TMP11:%.*]] = bitcast <16 x i32> [[TMP10]] to <8 x i64>
// CHECK-NEXT:    store <8 x i64> [[TMP11]], <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK-NEXT:    [[TMP12:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK-NEXT:    [[EXTRACT_I:%.*]] = shufflevector <8 x i64> [[TMP12]], <8 x i64> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// CHECK-NEXT:    store <4 x i64> [[EXTRACT_I]], <4 x i64>* [[__T1_I]], align 32
// CHECK-NEXT:    [[TMP13:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK-NEXT:    [[EXTRACT3_I:%.*]] = shufflevector <8 x i64> [[TMP13]], <8 x i64> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK-NEXT:    store <4 x i64> [[EXTRACT3_I]], <4 x i64>* [[__T2_I]], align 32
// CHECK-NEXT:    [[TMP14:%.*]] = load <4 x i64>, <4 x i64>* [[__T1_I]], align 32
// CHECK-NEXT:    [[TMP15:%.*]] = load <4 x i64>, <4 x i64>* [[__T2_I]], align 32
// CHECK-NEXT:    store <4 x i64> [[TMP14]], <4 x i64>* [[__A2_ADDR_I_I]], align 32
// CHECK-NEXT:    store <4 x i64> [[TMP15]], <4 x i64>* [[__B_ADDR_I_I]], align 32
// CHECK-NEXT:    [[TMP16:%.*]] = load <4 x i64>, <4 x i64>* [[__A2_ADDR_I_I]], align 32
// CHECK-NEXT:    [[TMP17:%.*]] = bitcast <4 x i64> [[TMP16]] to <8 x i32>
// CHECK-NEXT:    [[TMP18:%.*]] = load <4 x i64>, <4 x i64>* [[__B_ADDR_I_I]], align 32
// CHECK-NEXT:    [[TMP19:%.*]] = bitcast <4 x i64> [[TMP18]] to <8 x i32>
// CHECK-NEXT:    [[TMP20:%.*]] = icmp ugt <8 x i32> [[TMP17]], [[TMP19]]
// CHECK-NEXT:    [[TMP21:%.*]] = select <8 x i1> [[TMP20]], <8 x i32> [[TMP17]], <8 x i32> [[TMP19]]
// CHECK-NEXT:    [[TMP22:%.*]] = bitcast <8 x i32> [[TMP21]] to <4 x i64>
// CHECK-NEXT:    store <4 x i64> [[TMP22]], <4 x i64>* [[__T3_I]], align 32
// CHECK-NEXT:    [[TMP23:%.*]] = load <4 x i64>, <4 x i64>* [[__T3_I]], align 32
// CHECK-NEXT:    [[EXTRACT5_I:%.*]] = shufflevector <4 x i64> [[TMP23]], <4 x i64> undef, <2 x i32> <i32 0, i32 1>
// CHECK-NEXT:    store <2 x i64> [[EXTRACT5_I]], <2 x i64>* [[__T4_I]], align 16
// CHECK-NEXT:    [[TMP24:%.*]] = load <4 x i64>, <4 x i64>* [[__T3_I]], align 32
// CHECK-NEXT:    [[EXTRACT6_I:%.*]] = shufflevector <4 x i64> [[TMP24]], <4 x i64> undef, <2 x i32> <i32 2, i32 3>
// CHECK-NEXT:    store <2 x i64> [[EXTRACT6_I]], <2 x i64>* [[__T5_I]], align 16
// CHECK-NEXT:    [[TMP25:%.*]] = load <2 x i64>, <2 x i64>* [[__T4_I]], align 16
// CHECK-NEXT:    [[TMP26:%.*]] = load <2 x i64>, <2 x i64>* [[__T5_I]], align 16
// CHECK-NEXT:    store <2 x i64> [[TMP25]], <2 x i64>* [[__V1_ADDR_I13_I]], align 16
// CHECK-NEXT:    store <2 x i64> [[TMP26]], <2 x i64>* [[__V2_ADDR_I14_I]], align 16
// CHECK-NEXT:    [[TMP27:%.*]] = load <2 x i64>, <2 x i64>* [[__V1_ADDR_I13_I]], align 16
// CHECK-NEXT:    [[TMP28:%.*]] = bitcast <2 x i64> [[TMP27]] to <4 x i32>
// CHECK-NEXT:    [[TMP29:%.*]] = load <2 x i64>, <2 x i64>* [[__V2_ADDR_I14_I]], align 16
// CHECK-NEXT:    [[TMP30:%.*]] = bitcast <2 x i64> [[TMP29]] to <4 x i32>
// CHECK-NEXT:    [[TMP31:%.*]] = icmp ugt <4 x i32> [[TMP28]], [[TMP30]]
// CHECK-NEXT:    [[TMP32:%.*]] = select <4 x i1> [[TMP31]], <4 x i32> [[TMP28]], <4 x i32> [[TMP30]]
// CHECK-NEXT:    [[TMP33:%.*]] = bitcast <4 x i32> [[TMP32]] to <2 x i64>
// CHECK-NEXT:    store <2 x i64> [[TMP33]], <2 x i64>* [[__T6_I]], align 16
// CHECK-NEXT:    [[TMP34:%.*]] = load <2 x i64>, <2 x i64>* [[__T6_I]], align 16
// CHECK-NEXT:    [[TMP35:%.*]] = bitcast <2 x i64> [[TMP34]] to <4 x i32>
// CHECK-NEXT:    [[TMP36:%.*]] = load <2 x i64>, <2 x i64>* [[__T6_I]], align 16
// CHECK-NEXT:    [[TMP37:%.*]] = bitcast <2 x i64> [[TMP36]] to <4 x i32>
// CHECK-NEXT:    [[SHUFFLE_I:%.*]] = shufflevector <4 x i32> [[TMP35]], <4 x i32> [[TMP37]], <4 x i32> <i32 2, i32 3, i32 0, i32 1>
// CHECK-NEXT:    [[TMP38:%.*]] = bitcast <4 x i32> [[SHUFFLE_I]] to <2 x i64>
// CHECK-NEXT:    store <2 x i64> [[TMP38]], <2 x i64>* [[__T7_I]], align 16
// CHECK-NEXT:    [[TMP39:%.*]] = load <2 x i64>, <2 x i64>* [[__T6_I]], align 16
// CHECK-NEXT:    [[TMP40:%.*]] = load <2 x i64>, <2 x i64>* [[__T7_I]], align 16
// CHECK-NEXT:    store <2 x i64> [[TMP39]], <2 x i64>* [[__V1_ADDR_I11_I]], align 16
// CHECK-NEXT:    store <2 x i64> [[TMP40]], <2 x i64>* [[__V2_ADDR_I12_I]], align 16
// CHECK-NEXT:    [[TMP41:%.*]] = load <2 x i64>, <2 x i64>* [[__V1_ADDR_I11_I]], align 16
// CHECK-NEXT:    [[TMP42:%.*]] = bitcast <2 x i64> [[TMP41]] to <4 x i32>
// CHECK-NEXT:    [[TMP43:%.*]] = load <2 x i64>, <2 x i64>* [[__V2_ADDR_I12_I]], align 16
// CHECK-NEXT:    [[TMP44:%.*]] = bitcast <2 x i64> [[TMP43]] to <4 x i32>
// CHECK-NEXT:    [[TMP45:%.*]] = icmp ugt <4 x i32> [[TMP42]], [[TMP44]]
// CHECK-NEXT:    [[TMP46:%.*]] = select <4 x i1> [[TMP45]], <4 x i32> [[TMP42]], <4 x i32> [[TMP44]]
// CHECK-NEXT:    [[TMP47:%.*]] = bitcast <4 x i32> [[TMP46]] to <2 x i64>
// CHECK-NEXT:    store <2 x i64> [[TMP47]], <2 x i64>* [[__T8_I]], align 16
// CHECK-NEXT:    [[TMP48:%.*]] = load <2 x i64>, <2 x i64>* [[__T8_I]], align 16
// CHECK-NEXT:    [[TMP49:%.*]] = bitcast <2 x i64> [[TMP48]] to <4 x i32>
// CHECK-NEXT:    [[TMP50:%.*]] = load <2 x i64>, <2 x i64>* [[__T8_I]], align 16
// CHECK-NEXT:    [[TMP51:%.*]] = bitcast <2 x i64> [[TMP50]] to <4 x i32>
// CHECK-NEXT:    [[SHUFFLE9_I:%.*]] = shufflevector <4 x i32> [[TMP49]], <4 x i32> [[TMP51]], <4 x i32> <i32 1, i32 0, i32 3, i32 2>
// CHECK-NEXT:    [[TMP52:%.*]] = bitcast <4 x i32> [[SHUFFLE9_I]] to <2 x i64>
// CHECK-NEXT:    store <2 x i64> [[TMP52]], <2 x i64>* [[__T9_I]], align 16
// CHECK-NEXT:    [[TMP53:%.*]] = load <2 x i64>, <2 x i64>* [[__T8_I]], align 16
// CHECK-NEXT:    [[TMP54:%.*]] = load <2 x i64>, <2 x i64>* [[__T9_I]], align 16
// CHECK-NEXT:    store <2 x i64> [[TMP53]], <2 x i64>* [[__V1_ADDR_I_I]], align 16
// CHECK-NEXT:    store <2 x i64> [[TMP54]], <2 x i64>* [[__V2_ADDR_I_I]], align 16
// CHECK-NEXT:    [[TMP55:%.*]] = load <2 x i64>, <2 x i64>* [[__V1_ADDR_I_I]], align 16
// CHECK-NEXT:    [[TMP56:%.*]] = bitcast <2 x i64> [[TMP55]] to <4 x i32>
// CHECK-NEXT:    [[TMP57:%.*]] = load <2 x i64>, <2 x i64>* [[__V2_ADDR_I_I]], align 16
// CHECK-NEXT:    [[TMP58:%.*]] = bitcast <2 x i64> [[TMP57]] to <4 x i32>
// CHECK-NEXT:    [[TMP59:%.*]] = icmp ugt <4 x i32> [[TMP56]], [[TMP58]]
// CHECK-NEXT:    [[TMP60:%.*]] = select <4 x i1> [[TMP59]], <4 x i32> [[TMP56]], <4 x i32> [[TMP58]]
// CHECK-NEXT:    [[TMP61:%.*]] = bitcast <4 x i32> [[TMP60]] to <2 x i64>
// CHECK-NEXT:    store <4 x i32> [[TMP60]], <4 x i32>* [[__T10_I]], align 16
// CHECK-NEXT:    [[TMP62:%.*]] = load <4 x i32>, <4 x i32>* [[__T10_I]], align 16
// CHECK-NEXT:    [[VECEXT_I:%.*]] = extractelement <4 x i32> [[TMP62]], i32 0
// CHECK-NEXT:    ret i32 [[VECEXT_I]]
unsigned int test_mm512_mask_reduce_max_epu32(__mmask16 __M, __m512i __W){
  return _mm512_mask_reduce_max_epu32(__M, __W); 
}

// CHECK-LABEL: define float @test_mm512_mask_reduce_max_ps(i16 zeroext %__M, <16 x float> %__W) #0 {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[__W2_ADDR_I_I:%.*]] = alloca <16 x float>, align 64
// CHECK-NEXT:    [[__U_ADDR_I_I:%.*]] = alloca i16, align 2
// CHECK-NEXT:    [[__A_ADDR_I_I:%.*]] = alloca <16 x float>, align 64
// CHECK-NEXT:    [[__A_ADDR_I16_I:%.*]] = alloca <8 x float>, align 32
// CHECK-NEXT:    [[__B_ADDR_I17_I:%.*]] = alloca <8 x float>, align 32
// CHECK-NEXT:    [[__A_ADDR_I14_I:%.*]] = alloca <4 x float>, align 16
// CHECK-NEXT:    [[__B_ADDR_I15_I:%.*]] = alloca <4 x float>, align 16
// CHECK-NEXT:    [[__A_ADDR_I12_I:%.*]] = alloca <4 x float>, align 16
// CHECK-NEXT:    [[__B_ADDR_I13_I:%.*]] = alloca <4 x float>, align 16
// CHECK-NEXT:    [[__A2_ADDR_I_I:%.*]] = alloca <4 x float>, align 16
// CHECK-NEXT:    [[__B_ADDR_I_I:%.*]] = alloca <4 x float>, align 16
// CHECK-NEXT:    [[__W_ADDR_I_I:%.*]] = alloca float, align 4
// CHECK-NEXT:    [[DOTCOMPOUNDLITERAL_I_I:%.*]] = alloca <16 x float>, align 64
// CHECK-NEXT:    [[__M_ADDR_I:%.*]] = alloca i16, align 2
// CHECK-NEXT:    [[__V_ADDR_I:%.*]] = alloca <16 x float>, align 64
// CHECK-NEXT:    [[__T1_I:%.*]] = alloca <8 x float>, align 32
// CHECK-NEXT:    [[__T2_I:%.*]] = alloca <8 x float>, align 32
// CHECK-NEXT:    [[__T3_I:%.*]] = alloca <8 x float>, align 32
// CHECK-NEXT:    [[__T4_I:%.*]] = alloca <4 x float>, align 16
// CHECK-NEXT:    [[__T5_I:%.*]] = alloca <4 x float>, align 16
// CHECK-NEXT:    [[__T6_I:%.*]] = alloca <4 x float>, align 16
// CHECK-NEXT:    [[__T7_I:%.*]] = alloca <4 x float>, align 16
// CHECK-NEXT:    [[__T8_I:%.*]] = alloca <4 x float>, align 16
// CHECK-NEXT:    [[__T9_I:%.*]] = alloca <4 x float>, align 16
// CHECK-NEXT:    [[__T10_I:%.*]] = alloca <4 x float>, align 16
// CHECK-NEXT:    [[__M_ADDR:%.*]] = alloca i16, align 2
// CHECK-NEXT:    [[__W_ADDR:%.*]] = alloca <16 x float>, align 64
// CHECK-NEXT:    store i16 [[__M:%.*]], i16* [[__M_ADDR]], align 2
// CHECK-NEXT:    store <16 x float> [[__W:%.*]], <16 x float>* [[__W_ADDR]], align 64
// CHECK-NEXT:    [[TMP0:%.*]] = load i16, i16* [[__M_ADDR]], align 2
// CHECK-NEXT:    [[TMP1:%.*]] = load <16 x float>, <16 x float>* [[__W_ADDR]], align 64
// CHECK-NEXT:    store i16 [[TMP0]], i16* [[__M_ADDR_I]], align 2
// CHECK-NEXT:    store <16 x float> [[TMP1]], <16 x float>* [[__V_ADDR_I]], align 64
// CHECK-NEXT:    store float 0xFFF0000000000000, float* [[__W_ADDR_I_I]], align 4
// CHECK-NEXT:    [[TMP2:%.*]] = load float, float* [[__W_ADDR_I_I]], align 4
// CHECK-NEXT:    [[VECINIT_I_I:%.*]] = insertelement <16 x float> undef, float [[TMP2]], i32 0
// CHECK-NEXT:    [[TMP3:%.*]] = load float, float* [[__W_ADDR_I_I]], align 4
// CHECK-NEXT:    [[VECINIT1_I_I:%.*]] = insertelement <16 x float> [[VECINIT_I_I]], float [[TMP3]], i32 1
// CHECK-NEXT:    [[TMP4:%.*]] = load float, float* [[__W_ADDR_I_I]], align 4
// CHECK-NEXT:    [[VECINIT2_I_I:%.*]] = insertelement <16 x float> [[VECINIT1_I_I]], float [[TMP4]], i32 2
// CHECK-NEXT:    [[TMP5:%.*]] = load float, float* [[__W_ADDR_I_I]], align 4
// CHECK-NEXT:    [[VECINIT3_I_I:%.*]] = insertelement <16 x float> [[VECINIT2_I_I]], float [[TMP5]], i32 3
// CHECK-NEXT:    [[TMP6:%.*]] = load float, float* [[__W_ADDR_I_I]], align 4
// CHECK-NEXT:    [[VECINIT4_I_I:%.*]] = insertelement <16 x float> [[VECINIT3_I_I]], float [[TMP6]], i32 4
// CHECK-NEXT:    [[TMP7:%.*]] = load float, float* [[__W_ADDR_I_I]], align 4
// CHECK-NEXT:    [[VECINIT5_I_I:%.*]] = insertelement <16 x float> [[VECINIT4_I_I]], float [[TMP7]], i32 5
// CHECK-NEXT:    [[TMP8:%.*]] = load float, float* [[__W_ADDR_I_I]], align 4
// CHECK-NEXT:    [[VECINIT6_I_I:%.*]] = insertelement <16 x float> [[VECINIT5_I_I]], float [[TMP8]], i32 6
// CHECK-NEXT:    [[TMP9:%.*]] = load float, float* [[__W_ADDR_I_I]], align 4
// CHECK-NEXT:    [[VECINIT7_I_I:%.*]] = insertelement <16 x float> [[VECINIT6_I_I]], float [[TMP9]], i32 7
// CHECK-NEXT:    [[TMP10:%.*]] = load float, float* [[__W_ADDR_I_I]], align 4
// CHECK-NEXT:    [[VECINIT8_I_I:%.*]] = insertelement <16 x float> [[VECINIT7_I_I]], float [[TMP10]], i32 8
// CHECK-NEXT:    [[TMP11:%.*]] = load float, float* [[__W_ADDR_I_I]], align 4
// CHECK-NEXT:    [[VECINIT9_I_I:%.*]] = insertelement <16 x float> [[VECINIT8_I_I]], float [[TMP11]], i32 9
// CHECK-NEXT:    [[TMP12:%.*]] = load float, float* [[__W_ADDR_I_I]], align 4
// CHECK-NEXT:    [[VECINIT10_I_I:%.*]] = insertelement <16 x float> [[VECINIT9_I_I]], float [[TMP12]], i32 10
// CHECK-NEXT:    [[TMP13:%.*]] = load float, float* [[__W_ADDR_I_I]], align 4
// CHECK-NEXT:    [[VECINIT11_I_I:%.*]] = insertelement <16 x float> [[VECINIT10_I_I]], float [[TMP13]], i32 11
// CHECK-NEXT:    [[TMP14:%.*]] = load float, float* [[__W_ADDR_I_I]], align 4
// CHECK-NEXT:    [[VECINIT12_I_I:%.*]] = insertelement <16 x float> [[VECINIT11_I_I]], float [[TMP14]], i32 12
// CHECK-NEXT:    [[TMP15:%.*]] = load float, float* [[__W_ADDR_I_I]], align 4
// CHECK-NEXT:    [[VECINIT13_I_I:%.*]] = insertelement <16 x float> [[VECINIT12_I_I]], float [[TMP15]], i32 13
// CHECK-NEXT:    [[TMP16:%.*]] = load float, float* [[__W_ADDR_I_I]], align 4
// CHECK-NEXT:    [[VECINIT14_I_I:%.*]] = insertelement <16 x float> [[VECINIT13_I_I]], float [[TMP16]], i32 14
// CHECK-NEXT:    [[TMP17:%.*]] = load float, float* [[__W_ADDR_I_I]], align 4
// CHECK-NEXT:    [[VECINIT15_I_I:%.*]] = insertelement <16 x float> [[VECINIT14_I_I]], float [[TMP17]], i32 15
// CHECK-NEXT:    store <16 x float> [[VECINIT15_I_I]], <16 x float>* [[DOTCOMPOUNDLITERAL_I_I]], align 64
// CHECK-NEXT:    [[TMP18:%.*]] = load <16 x float>, <16 x float>* [[DOTCOMPOUNDLITERAL_I_I]], align 64
// CHECK-NEXT:    [[TMP19:%.*]] = load i16, i16* [[__M_ADDR_I]], align 2
// CHECK-NEXT:    [[TMP20:%.*]] = load <16 x float>, <16 x float>* [[__V_ADDR_I]], align 64
// CHECK-NEXT:    store <16 x float> [[TMP18]], <16 x float>* [[__W2_ADDR_I_I]], align 64
// CHECK-NEXT:    store i16 [[TMP19]], i16* [[__U_ADDR_I_I]], align 2
// CHECK-NEXT:    store <16 x float> [[TMP20]], <16 x float>* [[__A_ADDR_I_I]], align 64
// CHECK-NEXT:    [[TMP21:%.*]] = load i16, i16* [[__U_ADDR_I_I]], align 2
// CHECK-NEXT:    [[TMP22:%.*]] = load <16 x float>, <16 x float>* [[__A_ADDR_I_I]], align 64
// CHECK-NEXT:    [[TMP23:%.*]] = load <16 x float>, <16 x float>* [[__W2_ADDR_I_I]], align 64
// CHECK-NEXT:    [[TMP24:%.*]] = bitcast i16 [[TMP21]] to <16 x i1>
// CHECK-NEXT:    [[TMP25:%.*]] = select <16 x i1> [[TMP24]], <16 x float> [[TMP22]], <16 x float> [[TMP23]]
// CHECK-NEXT:    store <16 x float> [[TMP25]], <16 x float>* [[__V_ADDR_I]], align 64
// CHECK-NEXT:    [[TMP26:%.*]] = load <16 x float>, <16 x float>* [[__V_ADDR_I]], align 64
// CHECK-NEXT:    [[TMP27:%.*]] = bitcast <16 x float> [[TMP26]] to <8 x double>
// CHECK-NEXT:    [[EXTRACT_I:%.*]] = shufflevector <8 x double> [[TMP27]], <8 x double> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// CHECK-NEXT:    [[TMP28:%.*]] = bitcast <4 x double> [[EXTRACT_I]] to <8 x float>
// CHECK-NEXT:    store <8 x float> [[TMP28]], <8 x float>* [[__T1_I]], align 32
// CHECK-NEXT:    [[TMP29:%.*]] = load <16 x float>, <16 x float>* [[__V_ADDR_I]], align 64
// CHECK-NEXT:    [[TMP30:%.*]] = bitcast <16 x float> [[TMP29]] to <8 x double>
// CHECK-NEXT:    [[EXTRACT4_I:%.*]] = shufflevector <8 x double> [[TMP30]], <8 x double> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK-NEXT:    [[TMP31:%.*]] = bitcast <4 x double> [[EXTRACT4_I]] to <8 x float>
// CHECK-NEXT:    store <8 x float> [[TMP31]], <8 x float>* [[__T2_I]], align 32
// CHECK-NEXT:    [[TMP32:%.*]] = load <8 x float>, <8 x float>* [[__T1_I]], align 32
// CHECK-NEXT:    [[TMP33:%.*]] = load <8 x float>, <8 x float>* [[__T2_I]], align 32
// CHECK-NEXT:    store <8 x float> [[TMP32]], <8 x float>* [[__A_ADDR_I16_I]], align 32
// CHECK-NEXT:    store <8 x float> [[TMP33]], <8 x float>* [[__B_ADDR_I17_I]], align 32
// CHECK-NEXT:    [[TMP34:%.*]] = load <8 x float>, <8 x float>* [[__A_ADDR_I16_I]], align 32
// CHECK-NEXT:    [[TMP35:%.*]] = load <8 x float>, <8 x float>* [[__B_ADDR_I17_I]], align 32
// CHECK-NEXT:    [[TMP36:%.*]] = call <8 x float> @llvm.x86.avx.max.ps.256(<8 x float> [[TMP34]], <8 x float> [[TMP35]]) #2
// CHECK-NEXT:    store <8 x float> [[TMP36]], <8 x float>* [[__T3_I]], align 32
// CHECK-NEXT:    [[TMP37:%.*]] = load <8 x float>, <8 x float>* [[__T3_I]], align 32
// CHECK-NEXT:    [[EXTRACT6_I:%.*]] = shufflevector <8 x float> [[TMP37]], <8 x float> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// CHECK-NEXT:    store <4 x float> [[EXTRACT6_I]], <4 x float>* [[__T4_I]], align 16
// CHECK-NEXT:    [[TMP38:%.*]] = load <8 x float>, <8 x float>* [[__T3_I]], align 32
// CHECK-NEXT:    [[EXTRACT7_I:%.*]] = shufflevector <8 x float> [[TMP38]], <8 x float> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK-NEXT:    store <4 x float> [[EXTRACT7_I]], <4 x float>* [[__T5_I]], align 16
// CHECK-NEXT:    [[TMP39:%.*]] = load <4 x float>, <4 x float>* [[__T4_I]], align 16
// CHECK-NEXT:    [[TMP40:%.*]] = load <4 x float>, <4 x float>* [[__T5_I]], align 16
// CHECK-NEXT:    store <4 x float> [[TMP39]], <4 x float>* [[__A_ADDR_I14_I]], align 16
// CHECK-NEXT:    store <4 x float> [[TMP40]], <4 x float>* [[__B_ADDR_I15_I]], align 16
// CHECK-NEXT:    [[TMP41:%.*]] = load <4 x float>, <4 x float>* [[__A_ADDR_I14_I]], align 16
// CHECK-NEXT:    [[TMP42:%.*]] = load <4 x float>, <4 x float>* [[__B_ADDR_I15_I]], align 16
// CHECK-NEXT:    [[TMP43:%.*]] = call <4 x float> @llvm.x86.sse.max.ps(<4 x float> [[TMP41]], <4 x float> [[TMP42]]) #2
// CHECK-NEXT:    store <4 x float> [[TMP43]], <4 x float>* [[__T6_I]], align 16
// CHECK-NEXT:    [[TMP44:%.*]] = load <4 x float>, <4 x float>* [[__T6_I]], align 16
// CHECK-NEXT:    [[TMP45:%.*]] = load <4 x float>, <4 x float>* [[__T6_I]], align 16
// CHECK-NEXT:    [[SHUFFLE_I:%.*]] = shufflevector <4 x float> [[TMP44]], <4 x float> [[TMP45]], <4 x i32> <i32 2, i32 3, i32 0, i32 1>
// CHECK-NEXT:    store <4 x float> [[SHUFFLE_I]], <4 x float>* [[__T7_I]], align 16
// CHECK-NEXT:    [[TMP46:%.*]] = load <4 x float>, <4 x float>* [[__T6_I]], align 16
// CHECK-NEXT:    [[TMP47:%.*]] = load <4 x float>, <4 x float>* [[__T7_I]], align 16
// CHECK-NEXT:    store <4 x float> [[TMP46]], <4 x float>* [[__A_ADDR_I12_I]], align 16
// CHECK-NEXT:    store <4 x float> [[TMP47]], <4 x float>* [[__B_ADDR_I13_I]], align 16
// CHECK-NEXT:    [[TMP48:%.*]] = load <4 x float>, <4 x float>* [[__A_ADDR_I12_I]], align 16
// CHECK-NEXT:    [[TMP49:%.*]] = load <4 x float>, <4 x float>* [[__B_ADDR_I13_I]], align 16
// CHECK-NEXT:    [[TMP50:%.*]] = call <4 x float> @llvm.x86.sse.max.ps(<4 x float> [[TMP48]], <4 x float> [[TMP49]]) #2
// CHECK-NEXT:    store <4 x float> [[TMP50]], <4 x float>* [[__T8_I]], align 16
// CHECK-NEXT:    [[TMP51:%.*]] = load <4 x float>, <4 x float>* [[__T8_I]], align 16
// CHECK-NEXT:    [[TMP52:%.*]] = load <4 x float>, <4 x float>* [[__T8_I]], align 16
// CHECK-NEXT:    [[SHUFFLE10_I:%.*]] = shufflevector <4 x float> [[TMP51]], <4 x float> [[TMP52]], <4 x i32> <i32 1, i32 0, i32 3, i32 2>
// CHECK-NEXT:    store <4 x float> [[SHUFFLE10_I]], <4 x float>* [[__T9_I]], align 16
// CHECK-NEXT:    [[TMP53:%.*]] = load <4 x float>, <4 x float>* [[__T8_I]], align 16
// CHECK-NEXT:    [[TMP54:%.*]] = load <4 x float>, <4 x float>* [[__T9_I]], align 16
// CHECK-NEXT:    store <4 x float> [[TMP53]], <4 x float>* [[__A2_ADDR_I_I]], align 16
// CHECK-NEXT:    store <4 x float> [[TMP54]], <4 x float>* [[__B_ADDR_I_I]], align 16
// CHECK-NEXT:    [[TMP55:%.*]] = load <4 x float>, <4 x float>* [[__A2_ADDR_I_I]], align 16
// CHECK-NEXT:    [[TMP56:%.*]] = load <4 x float>, <4 x float>* [[__B_ADDR_I_I]], align 16
// CHECK-NEXT:    [[TMP57:%.*]] = call <4 x float> @llvm.x86.sse.max.ps(<4 x float> [[TMP55]], <4 x float> [[TMP56]]) #2
// CHECK-NEXT:    store <4 x float> [[TMP57]], <4 x float>* [[__T10_I]], align 16
// CHECK-NEXT:    [[TMP58:%.*]] = load <4 x float>, <4 x float>* [[__T10_I]], align 16
// CHECK-NEXT:    [[VECEXT_I:%.*]] = extractelement <4 x float> [[TMP58]], i32 0
// CHECK-NEXT:    ret float [[VECEXT_I]]
float test_mm512_mask_reduce_max_ps(__mmask16 __M, __m512 __W){
  return _mm512_mask_reduce_max_ps(__M, __W); 
}

// CHECK-LABEL: define i32 @test_mm512_mask_reduce_min_epi32(i16 zeroext %__M, <8 x i64> %__W) #0 {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[__W_ADDR_I_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__U_ADDR_I_I:%.*]] = alloca i16, align 2
// CHECK-NEXT:    [[__A_ADDR_I_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__A2_ADDR_I_I:%.*]] = alloca <4 x i64>, align 32
// CHECK-NEXT:    [[__B_ADDR_I_I:%.*]] = alloca <4 x i64>, align 32
// CHECK-NEXT:    [[__V1_ADDR_I14_I:%.*]] = alloca <2 x i64>, align 16
// CHECK-NEXT:    [[__V2_ADDR_I15_I:%.*]] = alloca <2 x i64>, align 16
// CHECK-NEXT:    [[__V1_ADDR_I12_I:%.*]] = alloca <2 x i64>, align 16
// CHECK-NEXT:    [[__V2_ADDR_I13_I:%.*]] = alloca <2 x i64>, align 16
// CHECK-NEXT:    [[__V1_ADDR_I_I:%.*]] = alloca <2 x i64>, align 16
// CHECK-NEXT:    [[__V2_ADDR_I_I:%.*]] = alloca <2 x i64>, align 16
// CHECK-NEXT:    [[__S_ADDR_I_I:%.*]] = alloca i32, align 4
// CHECK-NEXT:    [[DOTCOMPOUNDLITERAL_I_I:%.*]] = alloca <16 x i32>, align 64
// CHECK-NEXT:    [[__M_ADDR_I:%.*]] = alloca i16, align 2
// CHECK-NEXT:    [[__V_ADDR_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__T1_I:%.*]] = alloca <4 x i64>, align 32
// CHECK-NEXT:    [[__T2_I:%.*]] = alloca <4 x i64>, align 32
// CHECK-NEXT:    [[__T3_I:%.*]] = alloca <4 x i64>, align 32
// CHECK-NEXT:    [[__T4_I:%.*]] = alloca <2 x i64>, align 16
// CHECK-NEXT:    [[__T5_I:%.*]] = alloca <2 x i64>, align 16
// CHECK-NEXT:    [[__T6_I:%.*]] = alloca <2 x i64>, align 16
// CHECK-NEXT:    [[__T7_I:%.*]] = alloca <2 x i64>, align 16
// CHECK-NEXT:    [[__T8_I:%.*]] = alloca <2 x i64>, align 16
// CHECK-NEXT:    [[__T9_I:%.*]] = alloca <2 x i64>, align 16
// CHECK-NEXT:    [[__T10_I:%.*]] = alloca <4 x i32>, align 16
// CHECK-NEXT:    [[__M_ADDR:%.*]] = alloca i16, align 2
// CHECK-NEXT:    [[__W_ADDR:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    store i16 [[__M:%.*]], i16* [[__M_ADDR]], align 2
// CHECK-NEXT:    store <8 x i64> [[__W:%.*]], <8 x i64>* [[__W_ADDR]], align 64
// CHECK-NEXT:    [[TMP0:%.*]] = load i16, i16* [[__M_ADDR]], align 2
// CHECK-NEXT:    [[TMP1:%.*]] = load <8 x i64>, <8 x i64>* [[__W_ADDR]], align 64
// CHECK-NEXT:    store i16 [[TMP0]], i16* [[__M_ADDR_I]], align 2
// CHECK-NEXT:    store <8 x i64> [[TMP1]], <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK-NEXT:    store i32 2147483647, i32* [[__S_ADDR_I_I]], align 4
// CHECK-NEXT:    [[TMP2:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK-NEXT:    [[VECINIT_I_I:%.*]] = insertelement <16 x i32> undef, i32 [[TMP2]], i32 0
// CHECK-NEXT:    [[TMP3:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK-NEXT:    [[VECINIT1_I_I:%.*]] = insertelement <16 x i32> [[VECINIT_I_I]], i32 [[TMP3]], i32 1
// CHECK-NEXT:    [[TMP4:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK-NEXT:    [[VECINIT2_I_I:%.*]] = insertelement <16 x i32> [[VECINIT1_I_I]], i32 [[TMP4]], i32 2
// CHECK-NEXT:    [[TMP5:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK-NEXT:    [[VECINIT3_I_I:%.*]] = insertelement <16 x i32> [[VECINIT2_I_I]], i32 [[TMP5]], i32 3
// CHECK-NEXT:    [[TMP6:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK-NEXT:    [[VECINIT4_I_I:%.*]] = insertelement <16 x i32> [[VECINIT3_I_I]], i32 [[TMP6]], i32 4
// CHECK-NEXT:    [[TMP7:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK-NEXT:    [[VECINIT5_I_I:%.*]] = insertelement <16 x i32> [[VECINIT4_I_I]], i32 [[TMP7]], i32 5
// CHECK-NEXT:    [[TMP8:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK-NEXT:    [[VECINIT6_I_I:%.*]] = insertelement <16 x i32> [[VECINIT5_I_I]], i32 [[TMP8]], i32 6
// CHECK-NEXT:    [[TMP9:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK-NEXT:    [[VECINIT7_I_I:%.*]] = insertelement <16 x i32> [[VECINIT6_I_I]], i32 [[TMP9]], i32 7
// CHECK-NEXT:    [[TMP10:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK-NEXT:    [[VECINIT8_I_I:%.*]] = insertelement <16 x i32> [[VECINIT7_I_I]], i32 [[TMP10]], i32 8
// CHECK-NEXT:    [[TMP11:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK-NEXT:    [[VECINIT9_I_I:%.*]] = insertelement <16 x i32> [[VECINIT8_I_I]], i32 [[TMP11]], i32 9
// CHECK-NEXT:    [[TMP12:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK-NEXT:    [[VECINIT10_I_I:%.*]] = insertelement <16 x i32> [[VECINIT9_I_I]], i32 [[TMP12]], i32 10
// CHECK-NEXT:    [[TMP13:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK-NEXT:    [[VECINIT11_I_I:%.*]] = insertelement <16 x i32> [[VECINIT10_I_I]], i32 [[TMP13]], i32 11
// CHECK-NEXT:    [[TMP14:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK-NEXT:    [[VECINIT12_I_I:%.*]] = insertelement <16 x i32> [[VECINIT11_I_I]], i32 [[TMP14]], i32 12
// CHECK-NEXT:    [[TMP15:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK-NEXT:    [[VECINIT13_I_I:%.*]] = insertelement <16 x i32> [[VECINIT12_I_I]], i32 [[TMP15]], i32 13
// CHECK-NEXT:    [[TMP16:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK-NEXT:    [[VECINIT14_I_I:%.*]] = insertelement <16 x i32> [[VECINIT13_I_I]], i32 [[TMP16]], i32 14
// CHECK-NEXT:    [[TMP17:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK-NEXT:    [[VECINIT15_I_I:%.*]] = insertelement <16 x i32> [[VECINIT14_I_I]], i32 [[TMP17]], i32 15
// CHECK-NEXT:    store <16 x i32> [[VECINIT15_I_I]], <16 x i32>* [[DOTCOMPOUNDLITERAL_I_I]], align 64
// CHECK-NEXT:    [[TMP18:%.*]] = load <16 x i32>, <16 x i32>* [[DOTCOMPOUNDLITERAL_I_I]], align 64
// CHECK-NEXT:    [[TMP19:%.*]] = bitcast <16 x i32> [[TMP18]] to <8 x i64>
// CHECK-NEXT:    [[TMP20:%.*]] = load i16, i16* [[__M_ADDR_I]], align 2
// CHECK-NEXT:    [[TMP21:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK-NEXT:    store <8 x i64> [[TMP19]], <8 x i64>* [[__W_ADDR_I_I]], align 64
// CHECK-NEXT:    store i16 [[TMP20]], i16* [[__U_ADDR_I_I]], align 2
// CHECK-NEXT:    store <8 x i64> [[TMP21]], <8 x i64>* [[__A_ADDR_I_I]], align 64
// CHECK-NEXT:    [[TMP22:%.*]] = load i16, i16* [[__U_ADDR_I_I]], align 2
// CHECK-NEXT:    [[TMP23:%.*]] = load <8 x i64>, <8 x i64>* [[__A_ADDR_I_I]], align 64
// CHECK-NEXT:    [[TMP24:%.*]] = bitcast <8 x i64> [[TMP23]] to <16 x i32>
// CHECK-NEXT:    [[TMP25:%.*]] = load <8 x i64>, <8 x i64>* [[__W_ADDR_I_I]], align 64
// CHECK-NEXT:    [[TMP26:%.*]] = bitcast <8 x i64> [[TMP25]] to <16 x i32>
// CHECK-NEXT:    [[TMP27:%.*]] = bitcast i16 [[TMP22]] to <16 x i1>
// CHECK-NEXT:    [[TMP28:%.*]] = select <16 x i1> [[TMP27]], <16 x i32> [[TMP24]], <16 x i32> [[TMP26]]
// CHECK-NEXT:    [[TMP29:%.*]] = bitcast <16 x i32> [[TMP28]] to <8 x i64>
// CHECK-NEXT:    store <8 x i64> [[TMP29]], <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK-NEXT:    [[TMP30:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK-NEXT:    [[EXTRACT_I:%.*]] = shufflevector <8 x i64> [[TMP30]], <8 x i64> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// CHECK-NEXT:    store <4 x i64> [[EXTRACT_I]], <4 x i64>* [[__T1_I]], align 32
// CHECK-NEXT:    [[TMP31:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK-NEXT:    [[EXTRACT4_I:%.*]] = shufflevector <8 x i64> [[TMP31]], <8 x i64> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK-NEXT:    store <4 x i64> [[EXTRACT4_I]], <4 x i64>* [[__T2_I]], align 32
// CHECK-NEXT:    [[TMP32:%.*]] = load <4 x i64>, <4 x i64>* [[__T1_I]], align 32
// CHECK-NEXT:    [[TMP33:%.*]] = load <4 x i64>, <4 x i64>* [[__T2_I]], align 32
// CHECK-NEXT:    store <4 x i64> [[TMP32]], <4 x i64>* [[__A2_ADDR_I_I]], align 32
// CHECK-NEXT:    store <4 x i64> [[TMP33]], <4 x i64>* [[__B_ADDR_I_I]], align 32
// CHECK-NEXT:    [[TMP34:%.*]] = load <4 x i64>, <4 x i64>* [[__A2_ADDR_I_I]], align 32
// CHECK-NEXT:    [[TMP35:%.*]] = bitcast <4 x i64> [[TMP34]] to <8 x i32>
// CHECK-NEXT:    [[TMP36:%.*]] = load <4 x i64>, <4 x i64>* [[__B_ADDR_I_I]], align 32
// CHECK-NEXT:    [[TMP37:%.*]] = bitcast <4 x i64> [[TMP36]] to <8 x i32>
// CHECK-NEXT:    [[TMP38:%.*]] = icmp slt <8 x i32> [[TMP35]], [[TMP37]]
// CHECK-NEXT:    [[TMP39:%.*]] = select <8 x i1> [[TMP38]], <8 x i32> [[TMP35]], <8 x i32> [[TMP37]]
// CHECK-NEXT:    [[TMP40:%.*]] = bitcast <8 x i32> [[TMP39]] to <4 x i64>
// CHECK-NEXT:    store <4 x i64> [[TMP40]], <4 x i64>* [[__T3_I]], align 32
// CHECK-NEXT:    [[TMP41:%.*]] = load <4 x i64>, <4 x i64>* [[__T3_I]], align 32
// CHECK-NEXT:    [[EXTRACT6_I:%.*]] = shufflevector <4 x i64> [[TMP41]], <4 x i64> undef, <2 x i32> <i32 0, i32 1>
// CHECK-NEXT:    store <2 x i64> [[EXTRACT6_I]], <2 x i64>* [[__T4_I]], align 16
// CHECK-NEXT:    [[TMP42:%.*]] = load <4 x i64>, <4 x i64>* [[__T3_I]], align 32
// CHECK-NEXT:    [[EXTRACT7_I:%.*]] = shufflevector <4 x i64> [[TMP42]], <4 x i64> undef, <2 x i32> <i32 2, i32 3>
// CHECK-NEXT:    store <2 x i64> [[EXTRACT7_I]], <2 x i64>* [[__T5_I]], align 16
// CHECK-NEXT:    [[TMP43:%.*]] = load <2 x i64>, <2 x i64>* [[__T4_I]], align 16
// CHECK-NEXT:    [[TMP44:%.*]] = load <2 x i64>, <2 x i64>* [[__T5_I]], align 16
// CHECK-NEXT:    store <2 x i64> [[TMP43]], <2 x i64>* [[__V1_ADDR_I14_I]], align 16
// CHECK-NEXT:    store <2 x i64> [[TMP44]], <2 x i64>* [[__V2_ADDR_I15_I]], align 16
// CHECK-NEXT:    [[TMP45:%.*]] = load <2 x i64>, <2 x i64>* [[__V1_ADDR_I14_I]], align 16
// CHECK-NEXT:    [[TMP46:%.*]] = bitcast <2 x i64> [[TMP45]] to <4 x i32>
// CHECK-NEXT:    [[TMP47:%.*]] = load <2 x i64>, <2 x i64>* [[__V2_ADDR_I15_I]], align 16
// CHECK-NEXT:    [[TMP48:%.*]] = bitcast <2 x i64> [[TMP47]] to <4 x i32>
// CHECK-NEXT:    [[TMP49:%.*]] = icmp slt <4 x i32> [[TMP46]], [[TMP48]]
// CHECK-NEXT:    [[TMP50:%.*]] = select <4 x i1> [[TMP49]], <4 x i32> [[TMP46]], <4 x i32> [[TMP48]]
// CHECK-NEXT:    [[TMP51:%.*]] = bitcast <4 x i32> [[TMP50]] to <2 x i64>
// CHECK-NEXT:    store <2 x i64> [[TMP51]], <2 x i64>* [[__T6_I]], align 16
// CHECK-NEXT:    [[TMP52:%.*]] = load <2 x i64>, <2 x i64>* [[__T6_I]], align 16
// CHECK-NEXT:    [[TMP53:%.*]] = bitcast <2 x i64> [[TMP52]] to <4 x i32>
// CHECK-NEXT:    [[TMP54:%.*]] = load <2 x i64>, <2 x i64>* [[__T6_I]], align 16
// CHECK-NEXT:    [[TMP55:%.*]] = bitcast <2 x i64> [[TMP54]] to <4 x i32>
// CHECK-NEXT:    [[SHUFFLE_I:%.*]] = shufflevector <4 x i32> [[TMP53]], <4 x i32> [[TMP55]], <4 x i32> <i32 2, i32 3, i32 0, i32 1>
// CHECK-NEXT:    [[TMP56:%.*]] = bitcast <4 x i32> [[SHUFFLE_I]] to <2 x i64>
// CHECK-NEXT:    store <2 x i64> [[TMP56]], <2 x i64>* [[__T7_I]], align 16
// CHECK-NEXT:    [[TMP57:%.*]] = load <2 x i64>, <2 x i64>* [[__T6_I]], align 16
// CHECK-NEXT:    [[TMP58:%.*]] = load <2 x i64>, <2 x i64>* [[__T7_I]], align 16
// CHECK-NEXT:    store <2 x i64> [[TMP57]], <2 x i64>* [[__V1_ADDR_I12_I]], align 16
// CHECK-NEXT:    store <2 x i64> [[TMP58]], <2 x i64>* [[__V2_ADDR_I13_I]], align 16
// CHECK-NEXT:    [[TMP59:%.*]] = load <2 x i64>, <2 x i64>* [[__V1_ADDR_I12_I]], align 16
// CHECK-NEXT:    [[TMP60:%.*]] = bitcast <2 x i64> [[TMP59]] to <4 x i32>
// CHECK-NEXT:    [[TMP61:%.*]] = load <2 x i64>, <2 x i64>* [[__V2_ADDR_I13_I]], align 16
// CHECK-NEXT:    [[TMP62:%.*]] = bitcast <2 x i64> [[TMP61]] to <4 x i32>
// CHECK-NEXT:    [[TMP63:%.*]] = icmp slt <4 x i32> [[TMP60]], [[TMP62]]
// CHECK-NEXT:    [[TMP64:%.*]] = select <4 x i1> [[TMP63]], <4 x i32> [[TMP60]], <4 x i32> [[TMP62]]
// CHECK-NEXT:    [[TMP65:%.*]] = bitcast <4 x i32> [[TMP64]] to <2 x i64>
// CHECK-NEXT:    store <2 x i64> [[TMP65]], <2 x i64>* [[__T8_I]], align 16
// CHECK-NEXT:    [[TMP66:%.*]] = load <2 x i64>, <2 x i64>* [[__T8_I]], align 16
// CHECK-NEXT:    [[TMP67:%.*]] = bitcast <2 x i64> [[TMP66]] to <4 x i32>
// CHECK-NEXT:    [[TMP68:%.*]] = load <2 x i64>, <2 x i64>* [[__T8_I]], align 16
// CHECK-NEXT:    [[TMP69:%.*]] = bitcast <2 x i64> [[TMP68]] to <4 x i32>
// CHECK-NEXT:    [[SHUFFLE10_I:%.*]] = shufflevector <4 x i32> [[TMP67]], <4 x i32> [[TMP69]], <4 x i32> <i32 1, i32 0, i32 3, i32 2>
// CHECK-NEXT:    [[TMP70:%.*]] = bitcast <4 x i32> [[SHUFFLE10_I]] to <2 x i64>
// CHECK-NEXT:    store <2 x i64> [[TMP70]], <2 x i64>* [[__T9_I]], align 16
// CHECK-NEXT:    [[TMP71:%.*]] = load <2 x i64>, <2 x i64>* [[__T8_I]], align 16
// CHECK-NEXT:    [[TMP72:%.*]] = load <2 x i64>, <2 x i64>* [[__T9_I]], align 16
// CHECK-NEXT:    store <2 x i64> [[TMP71]], <2 x i64>* [[__V1_ADDR_I_I]], align 16
// CHECK-NEXT:    store <2 x i64> [[TMP72]], <2 x i64>* [[__V2_ADDR_I_I]], align 16
// CHECK-NEXT:    [[TMP73:%.*]] = load <2 x i64>, <2 x i64>* [[__V1_ADDR_I_I]], align 16
// CHECK-NEXT:    [[TMP74:%.*]] = bitcast <2 x i64> [[TMP73]] to <4 x i32>
// CHECK-NEXT:    [[TMP75:%.*]] = load <2 x i64>, <2 x i64>* [[__V2_ADDR_I_I]], align 16
// CHECK-NEXT:    [[TMP76:%.*]] = bitcast <2 x i64> [[TMP75]] to <4 x i32>
// CHECK-NEXT:    [[TMP77:%.*]] = icmp slt <4 x i32> [[TMP74]], [[TMP76]]
// CHECK-NEXT:    [[TMP78:%.*]] = select <4 x i1> [[TMP77]], <4 x i32> [[TMP74]], <4 x i32> [[TMP76]]
// CHECK-NEXT:    [[TMP79:%.*]] = bitcast <4 x i32> [[TMP78]] to <2 x i64>
// CHECK-NEXT:    store <4 x i32> [[TMP78]], <4 x i32>* [[__T10_I]], align 16
// CHECK-NEXT:    [[TMP80:%.*]] = load <4 x i32>, <4 x i32>* [[__T10_I]], align 16
// CHECK-NEXT:    [[VECEXT_I:%.*]] = extractelement <4 x i32> [[TMP80]], i32 0
// CHECK-NEXT:    ret i32 [[VECEXT_I]]
int test_mm512_mask_reduce_min_epi32(__mmask16 __M, __m512i __W){
  return _mm512_mask_reduce_min_epi32(__M, __W); 
}

// CHECK-LABEL: define i32 @test_mm512_mask_reduce_min_epu32(i16 zeroext %__M, <8 x i64> %__W) #0 {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[__W_ADDR_I_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__U_ADDR_I_I:%.*]] = alloca i16, align 2
// CHECK-NEXT:    [[__A_ADDR_I_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__A2_ADDR_I_I:%.*]] = alloca <4 x i64>, align 32
// CHECK-NEXT:    [[__B_ADDR_I_I:%.*]] = alloca <4 x i64>, align 32
// CHECK-NEXT:    [[__V1_ADDR_I14_I:%.*]] = alloca <2 x i64>, align 16
// CHECK-NEXT:    [[__V2_ADDR_I15_I:%.*]] = alloca <2 x i64>, align 16
// CHECK-NEXT:    [[__V1_ADDR_I12_I:%.*]] = alloca <2 x i64>, align 16
// CHECK-NEXT:    [[__V2_ADDR_I13_I:%.*]] = alloca <2 x i64>, align 16
// CHECK-NEXT:    [[__V1_ADDR_I_I:%.*]] = alloca <2 x i64>, align 16
// CHECK-NEXT:    [[__V2_ADDR_I_I:%.*]] = alloca <2 x i64>, align 16
// CHECK-NEXT:    [[__S_ADDR_I_I:%.*]] = alloca i32, align 4
// CHECK-NEXT:    [[DOTCOMPOUNDLITERAL_I_I:%.*]] = alloca <16 x i32>, align 64
// CHECK-NEXT:    [[__M_ADDR_I:%.*]] = alloca i16, align 2
// CHECK-NEXT:    [[__V_ADDR_I:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    [[__T1_I:%.*]] = alloca <4 x i64>, align 32
// CHECK-NEXT:    [[__T2_I:%.*]] = alloca <4 x i64>, align 32
// CHECK-NEXT:    [[__T3_I:%.*]] = alloca <4 x i64>, align 32
// CHECK-NEXT:    [[__T4_I:%.*]] = alloca <2 x i64>, align 16
// CHECK-NEXT:    [[__T5_I:%.*]] = alloca <2 x i64>, align 16
// CHECK-NEXT:    [[__T6_I:%.*]] = alloca <2 x i64>, align 16
// CHECK-NEXT:    [[__T7_I:%.*]] = alloca <2 x i64>, align 16
// CHECK-NEXT:    [[__T8_I:%.*]] = alloca <2 x i64>, align 16
// CHECK-NEXT:    [[__T9_I:%.*]] = alloca <2 x i64>, align 16
// CHECK-NEXT:    [[__T10_I:%.*]] = alloca <4 x i32>, align 16
// CHECK-NEXT:    [[__M_ADDR:%.*]] = alloca i16, align 2
// CHECK-NEXT:    [[__W_ADDR:%.*]] = alloca <8 x i64>, align 64
// CHECK-NEXT:    store i16 [[__M:%.*]], i16* [[__M_ADDR]], align 2
// CHECK-NEXT:    store <8 x i64> [[__W:%.*]], <8 x i64>* [[__W_ADDR]], align 64
// CHECK-NEXT:    [[TMP0:%.*]] = load i16, i16* [[__M_ADDR]], align 2
// CHECK-NEXT:    [[TMP1:%.*]] = load <8 x i64>, <8 x i64>* [[__W_ADDR]], align 64
// CHECK-NEXT:    store i16 [[TMP0]], i16* [[__M_ADDR_I]], align 2
// CHECK-NEXT:    store <8 x i64> [[TMP1]], <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK-NEXT:    store i32 -1, i32* [[__S_ADDR_I_I]], align 4
// CHECK-NEXT:    [[TMP2:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK-NEXT:    [[VECINIT_I_I:%.*]] = insertelement <16 x i32> undef, i32 [[TMP2]], i32 0
// CHECK-NEXT:    [[TMP3:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK-NEXT:    [[VECINIT1_I_I:%.*]] = insertelement <16 x i32> [[VECINIT_I_I]], i32 [[TMP3]], i32 1
// CHECK-NEXT:    [[TMP4:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK-NEXT:    [[VECINIT2_I_I:%.*]] = insertelement <16 x i32> [[VECINIT1_I_I]], i32 [[TMP4]], i32 2
// CHECK-NEXT:    [[TMP5:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK-NEXT:    [[VECINIT3_I_I:%.*]] = insertelement <16 x i32> [[VECINIT2_I_I]], i32 [[TMP5]], i32 3
// CHECK-NEXT:    [[TMP6:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK-NEXT:    [[VECINIT4_I_I:%.*]] = insertelement <16 x i32> [[VECINIT3_I_I]], i32 [[TMP6]], i32 4
// CHECK-NEXT:    [[TMP7:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK-NEXT:    [[VECINIT5_I_I:%.*]] = insertelement <16 x i32> [[VECINIT4_I_I]], i32 [[TMP7]], i32 5
// CHECK-NEXT:    [[TMP8:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK-NEXT:    [[VECINIT6_I_I:%.*]] = insertelement <16 x i32> [[VECINIT5_I_I]], i32 [[TMP8]], i32 6
// CHECK-NEXT:    [[TMP9:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK-NEXT:    [[VECINIT7_I_I:%.*]] = insertelement <16 x i32> [[VECINIT6_I_I]], i32 [[TMP9]], i32 7
// CHECK-NEXT:    [[TMP10:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK-NEXT:    [[VECINIT8_I_I:%.*]] = insertelement <16 x i32> [[VECINIT7_I_I]], i32 [[TMP10]], i32 8
// CHECK-NEXT:    [[TMP11:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK-NEXT:    [[VECINIT9_I_I:%.*]] = insertelement <16 x i32> [[VECINIT8_I_I]], i32 [[TMP11]], i32 9
// CHECK-NEXT:    [[TMP12:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK-NEXT:    [[VECINIT10_I_I:%.*]] = insertelement <16 x i32> [[VECINIT9_I_I]], i32 [[TMP12]], i32 10
// CHECK-NEXT:    [[TMP13:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK-NEXT:    [[VECINIT11_I_I:%.*]] = insertelement <16 x i32> [[VECINIT10_I_I]], i32 [[TMP13]], i32 11
// CHECK-NEXT:    [[TMP14:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK-NEXT:    [[VECINIT12_I_I:%.*]] = insertelement <16 x i32> [[VECINIT11_I_I]], i32 [[TMP14]], i32 12
// CHECK-NEXT:    [[TMP15:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK-NEXT:    [[VECINIT13_I_I:%.*]] = insertelement <16 x i32> [[VECINIT12_I_I]], i32 [[TMP15]], i32 13
// CHECK-NEXT:    [[TMP16:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK-NEXT:    [[VECINIT14_I_I:%.*]] = insertelement <16 x i32> [[VECINIT13_I_I]], i32 [[TMP16]], i32 14
// CHECK-NEXT:    [[TMP17:%.*]] = load i32, i32* [[__S_ADDR_I_I]], align 4
// CHECK-NEXT:    [[VECINIT15_I_I:%.*]] = insertelement <16 x i32> [[VECINIT14_I_I]], i32 [[TMP17]], i32 15
// CHECK-NEXT:    store <16 x i32> [[VECINIT15_I_I]], <16 x i32>* [[DOTCOMPOUNDLITERAL_I_I]], align 64
// CHECK-NEXT:    [[TMP18:%.*]] = load <16 x i32>, <16 x i32>* [[DOTCOMPOUNDLITERAL_I_I]], align 64
// CHECK-NEXT:    [[TMP19:%.*]] = bitcast <16 x i32> [[TMP18]] to <8 x i64>
// CHECK-NEXT:    [[TMP20:%.*]] = load i16, i16* [[__M_ADDR_I]], align 2
// CHECK-NEXT:    [[TMP21:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK-NEXT:    store <8 x i64> [[TMP19]], <8 x i64>* [[__W_ADDR_I_I]], align 64
// CHECK-NEXT:    store i16 [[TMP20]], i16* [[__U_ADDR_I_I]], align 2
// CHECK-NEXT:    store <8 x i64> [[TMP21]], <8 x i64>* [[__A_ADDR_I_I]], align 64
// CHECK-NEXT:    [[TMP22:%.*]] = load i16, i16* [[__U_ADDR_I_I]], align 2
// CHECK-NEXT:    [[TMP23:%.*]] = load <8 x i64>, <8 x i64>* [[__A_ADDR_I_I]], align 64
// CHECK-NEXT:    [[TMP24:%.*]] = bitcast <8 x i64> [[TMP23]] to <16 x i32>
// CHECK-NEXT:    [[TMP25:%.*]] = load <8 x i64>, <8 x i64>* [[__W_ADDR_I_I]], align 64
// CHECK-NEXT:    [[TMP26:%.*]] = bitcast <8 x i64> [[TMP25]] to <16 x i32>
// CHECK-NEXT:    [[TMP27:%.*]] = bitcast i16 [[TMP22]] to <16 x i1>
// CHECK-NEXT:    [[TMP28:%.*]] = select <16 x i1> [[TMP27]], <16 x i32> [[TMP24]], <16 x i32> [[TMP26]]
// CHECK-NEXT:    [[TMP29:%.*]] = bitcast <16 x i32> [[TMP28]] to <8 x i64>
// CHECK-NEXT:    store <8 x i64> [[TMP29]], <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK-NEXT:    [[TMP30:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK-NEXT:    [[EXTRACT_I:%.*]] = shufflevector <8 x i64> [[TMP30]], <8 x i64> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// CHECK-NEXT:    store <4 x i64> [[EXTRACT_I]], <4 x i64>* [[__T1_I]], align 32
// CHECK-NEXT:    [[TMP31:%.*]] = load <8 x i64>, <8 x i64>* [[__V_ADDR_I]], align 64
// CHECK-NEXT:    [[EXTRACT4_I:%.*]] = shufflevector <8 x i64> [[TMP31]], <8 x i64> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK-NEXT:    store <4 x i64> [[EXTRACT4_I]], <4 x i64>* [[__T2_I]], align 32
// CHECK-NEXT:    [[TMP32:%.*]] = load <4 x i64>, <4 x i64>* [[__T1_I]], align 32
// CHECK-NEXT:    [[TMP33:%.*]] = load <4 x i64>, <4 x i64>* [[__T2_I]], align 32
// CHECK-NEXT:    store <4 x i64> [[TMP32]], <4 x i64>* [[__A2_ADDR_I_I]], align 32
// CHECK-NEXT:    store <4 x i64> [[TMP33]], <4 x i64>* [[__B_ADDR_I_I]], align 32
// CHECK-NEXT:    [[TMP34:%.*]] = load <4 x i64>, <4 x i64>* [[__A2_ADDR_I_I]], align 32
// CHECK-NEXT:    [[TMP35:%.*]] = bitcast <4 x i64> [[TMP34]] to <8 x i32>
// CHECK-NEXT:    [[TMP36:%.*]] = load <4 x i64>, <4 x i64>* [[__B_ADDR_I_I]], align 32
// CHECK-NEXT:    [[TMP37:%.*]] = bitcast <4 x i64> [[TMP36]] to <8 x i32>
// CHECK-NEXT:    [[TMP38:%.*]] = icmp ult <8 x i32> [[TMP35]], [[TMP37]]
// CHECK-NEXT:    [[TMP39:%.*]] = select <8 x i1> [[TMP38]], <8 x i32> [[TMP35]], <8 x i32> [[TMP37]]
// CHECK-NEXT:    [[TMP40:%.*]] = bitcast <8 x i32> [[TMP39]] to <4 x i64>
// CHECK-NEXT:    store <4 x i64> [[TMP40]], <4 x i64>* [[__T3_I]], align 32
// CHECK-NEXT:    [[TMP41:%.*]] = load <4 x i64>, <4 x i64>* [[__T3_I]], align 32
// CHECK-NEXT:    [[EXTRACT6_I:%.*]] = shufflevector <4 x i64> [[TMP41]], <4 x i64> undef, <2 x i32> <i32 0, i32 1>
// CHECK-NEXT:    store <2 x i64> [[EXTRACT6_I]], <2 x i64>* [[__T4_I]], align 16
// CHECK-NEXT:    [[TMP42:%.*]] = load <4 x i64>, <4 x i64>* [[__T3_I]], align 32
// CHECK-NEXT:    [[EXTRACT7_I:%.*]] = shufflevector <4 x i64> [[TMP42]], <4 x i64> undef, <2 x i32> <i32 2, i32 3>
// CHECK-NEXT:    store <2 x i64> [[EXTRACT7_I]], <2 x i64>* [[__T5_I]], align 16
// CHECK-NEXT:    [[TMP43:%.*]] = load <2 x i64>, <2 x i64>* [[__T4_I]], align 16
// CHECK-NEXT:    [[TMP44:%.*]] = load <2 x i64>, <2 x i64>* [[__T5_I]], align 16
// CHECK-NEXT:    store <2 x i64> [[TMP43]], <2 x i64>* [[__V1_ADDR_I14_I]], align 16
// CHECK-NEXT:    store <2 x i64> [[TMP44]], <2 x i64>* [[__V2_ADDR_I15_I]], align 16
// CHECK-NEXT:    [[TMP45:%.*]] = load <2 x i64>, <2 x i64>* [[__V1_ADDR_I14_I]], align 16
// CHECK-NEXT:    [[TMP46:%.*]] = bitcast <2 x i64> [[TMP45]] to <4 x i32>
// CHECK-NEXT:    [[TMP47:%.*]] = load <2 x i64>, <2 x i64>* [[__V2_ADDR_I15_I]], align 16
// CHECK-NEXT:    [[TMP48:%.*]] = bitcast <2 x i64> [[TMP47]] to <4 x i32>
// CHECK-NEXT:    [[TMP49:%.*]] = icmp ult <4 x i32> [[TMP46]], [[TMP48]]
// CHECK-NEXT:    [[TMP50:%.*]] = select <4 x i1> [[TMP49]], <4 x i32> [[TMP46]], <4 x i32> [[TMP48]]
// CHECK-NEXT:    [[TMP51:%.*]] = bitcast <4 x i32> [[TMP50]] to <2 x i64>
// CHECK-NEXT:    store <2 x i64> [[TMP51]], <2 x i64>* [[__T6_I]], align 16
// CHECK-NEXT:    [[TMP52:%.*]] = load <2 x i64>, <2 x i64>* [[__T6_I]], align 16
// CHECK-NEXT:    [[TMP53:%.*]] = bitcast <2 x i64> [[TMP52]] to <4 x i32>
// CHECK-NEXT:    [[TMP54:%.*]] = load <2 x i64>, <2 x i64>* [[__T6_I]], align 16
// CHECK-NEXT:    [[TMP55:%.*]] = bitcast <2 x i64> [[TMP54]] to <4 x i32>
// CHECK-NEXT:    [[SHUFFLE_I:%.*]] = shufflevector <4 x i32> [[TMP53]], <4 x i32> [[TMP55]], <4 x i32> <i32 2, i32 3, i32 0, i32 1>
// CHECK-NEXT:    [[TMP56:%.*]] = bitcast <4 x i32> [[SHUFFLE_I]] to <2 x i64>
// CHECK-NEXT:    store <2 x i64> [[TMP56]], <2 x i64>* [[__T7_I]], align 16
// CHECK-NEXT:    [[TMP57:%.*]] = load <2 x i64>, <2 x i64>* [[__T6_I]], align 16
// CHECK-NEXT:    [[TMP58:%.*]] = load <2 x i64>, <2 x i64>* [[__T7_I]], align 16
// CHECK-NEXT:    store <2 x i64> [[TMP57]], <2 x i64>* [[__V1_ADDR_I12_I]], align 16
// CHECK-NEXT:    store <2 x i64> [[TMP58]], <2 x i64>* [[__V2_ADDR_I13_I]], align 16
// CHECK-NEXT:    [[TMP59:%.*]] = load <2 x i64>, <2 x i64>* [[__V1_ADDR_I12_I]], align 16
// CHECK-NEXT:    [[TMP60:%.*]] = bitcast <2 x i64> [[TMP59]] to <4 x i32>
// CHECK-NEXT:    [[TMP61:%.*]] = load <2 x i64>, <2 x i64>* [[__V2_ADDR_I13_I]], align 16
// CHECK-NEXT:    [[TMP62:%.*]] = bitcast <2 x i64> [[TMP61]] to <4 x i32>
// CHECK-NEXT:    [[TMP63:%.*]] = icmp ult <4 x i32> [[TMP60]], [[TMP62]]
// CHECK-NEXT:    [[TMP64:%.*]] = select <4 x i1> [[TMP63]], <4 x i32> [[TMP60]], <4 x i32> [[TMP62]]
// CHECK-NEXT:    [[TMP65:%.*]] = bitcast <4 x i32> [[TMP64]] to <2 x i64>
// CHECK-NEXT:    store <2 x i64> [[TMP65]], <2 x i64>* [[__T8_I]], align 16
// CHECK-NEXT:    [[TMP66:%.*]] = load <2 x i64>, <2 x i64>* [[__T8_I]], align 16
// CHECK-NEXT:    [[TMP67:%.*]] = bitcast <2 x i64> [[TMP66]] to <4 x i32>
// CHECK-NEXT:    [[TMP68:%.*]] = load <2 x i64>, <2 x i64>* [[__T8_I]], align 16
// CHECK-NEXT:    [[TMP69:%.*]] = bitcast <2 x i64> [[TMP68]] to <4 x i32>
// CHECK-NEXT:    [[SHUFFLE10_I:%.*]] = shufflevector <4 x i32> [[TMP67]], <4 x i32> [[TMP69]], <4 x i32> <i32 1, i32 0, i32 3, i32 2>
// CHECK-NEXT:    [[TMP70:%.*]] = bitcast <4 x i32> [[SHUFFLE10_I]] to <2 x i64>
// CHECK-NEXT:    store <2 x i64> [[TMP70]], <2 x i64>* [[__T9_I]], align 16
// CHECK-NEXT:    [[TMP71:%.*]] = load <2 x i64>, <2 x i64>* [[__T8_I]], align 16
// CHECK-NEXT:    [[TMP72:%.*]] = load <2 x i64>, <2 x i64>* [[__T9_I]], align 16
// CHECK-NEXT:    store <2 x i64> [[TMP71]], <2 x i64>* [[__V1_ADDR_I_I]], align 16
// CHECK-NEXT:    store <2 x i64> [[TMP72]], <2 x i64>* [[__V2_ADDR_I_I]], align 16
// CHECK-NEXT:    [[TMP73:%.*]] = load <2 x i64>, <2 x i64>* [[__V1_ADDR_I_I]], align 16
// CHECK-NEXT:    [[TMP74:%.*]] = bitcast <2 x i64> [[TMP73]] to <4 x i32>
// CHECK-NEXT:    [[TMP75:%.*]] = load <2 x i64>, <2 x i64>* [[__V2_ADDR_I_I]], align 16
// CHECK-NEXT:    [[TMP76:%.*]] = bitcast <2 x i64> [[TMP75]] to <4 x i32>
// CHECK-NEXT:    [[TMP77:%.*]] = icmp ult <4 x i32> [[TMP74]], [[TMP76]]
// CHECK-NEXT:    [[TMP78:%.*]] = select <4 x i1> [[TMP77]], <4 x i32> [[TMP74]], <4 x i32> [[TMP76]]
// CHECK-NEXT:    [[TMP79:%.*]] = bitcast <4 x i32> [[TMP78]] to <2 x i64>
// CHECK-NEXT:    store <4 x i32> [[TMP78]], <4 x i32>* [[__T10_I]], align 16
// CHECK-NEXT:    [[TMP80:%.*]] = load <4 x i32>, <4 x i32>* [[__T10_I]], align 16
// CHECK-NEXT:    [[VECEXT_I:%.*]] = extractelement <4 x i32> [[TMP80]], i32 0
// CHECK-NEXT:    ret i32 [[VECEXT_I]]
unsigned int test_mm512_mask_reduce_min_epu32(__mmask16 __M, __m512i __W){
  return _mm512_mask_reduce_min_epu32(__M, __W); 
}

// CHECK-LABEL: define float @test_mm512_mask_reduce_min_ps(i16 zeroext %__M, <16 x float> %__W) #0 {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[__W2_ADDR_I_I:%.*]] = alloca <16 x float>, align 64
// CHECK-NEXT:    [[__U_ADDR_I_I:%.*]] = alloca i16, align 2
// CHECK-NEXT:    [[__A_ADDR_I_I:%.*]] = alloca <16 x float>, align 64
// CHECK-NEXT:    [[__A_ADDR_I16_I:%.*]] = alloca <8 x float>, align 32
// CHECK-NEXT:    [[__B_ADDR_I17_I:%.*]] = alloca <8 x float>, align 32
// CHECK-NEXT:    [[__A_ADDR_I14_I:%.*]] = alloca <4 x float>, align 16
// CHECK-NEXT:    [[__B_ADDR_I15_I:%.*]] = alloca <4 x float>, align 16
// CHECK-NEXT:    [[__A_ADDR_I12_I:%.*]] = alloca <4 x float>, align 16
// CHECK-NEXT:    [[__B_ADDR_I13_I:%.*]] = alloca <4 x float>, align 16
// CHECK-NEXT:    [[__A2_ADDR_I_I:%.*]] = alloca <4 x float>, align 16
// CHECK-NEXT:    [[__B_ADDR_I_I:%.*]] = alloca <4 x float>, align 16
// CHECK-NEXT:    [[__W_ADDR_I_I:%.*]] = alloca float, align 4
// CHECK-NEXT:    [[DOTCOMPOUNDLITERAL_I_I:%.*]] = alloca <16 x float>, align 64
// CHECK-NEXT:    [[__M_ADDR_I:%.*]] = alloca i16, align 2
// CHECK-NEXT:    [[__V_ADDR_I:%.*]] = alloca <16 x float>, align 64
// CHECK-NEXT:    [[__T1_I:%.*]] = alloca <8 x float>, align 32
// CHECK-NEXT:    [[__T2_I:%.*]] = alloca <8 x float>, align 32
// CHECK-NEXT:    [[__T3_I:%.*]] = alloca <8 x float>, align 32
// CHECK-NEXT:    [[__T4_I:%.*]] = alloca <4 x float>, align 16
// CHECK-NEXT:    [[__T5_I:%.*]] = alloca <4 x float>, align 16
// CHECK-NEXT:    [[__T6_I:%.*]] = alloca <4 x float>, align 16
// CHECK-NEXT:    [[__T7_I:%.*]] = alloca <4 x float>, align 16
// CHECK-NEXT:    [[__T8_I:%.*]] = alloca <4 x float>, align 16
// CHECK-NEXT:    [[__T9_I:%.*]] = alloca <4 x float>, align 16
// CHECK-NEXT:    [[__T10_I:%.*]] = alloca <4 x float>, align 16
// CHECK-NEXT:    [[__M_ADDR:%.*]] = alloca i16, align 2
// CHECK-NEXT:    [[__W_ADDR:%.*]] = alloca <16 x float>, align 64
// CHECK-NEXT:    store i16 [[__M:%.*]], i16* [[__M_ADDR]], align 2
// CHECK-NEXT:    store <16 x float> [[__W:%.*]], <16 x float>* [[__W_ADDR]], align 64
// CHECK-NEXT:    [[TMP0:%.*]] = load i16, i16* [[__M_ADDR]], align 2
// CHECK-NEXT:    [[TMP1:%.*]] = load <16 x float>, <16 x float>* [[__W_ADDR]], align 64
// CHECK-NEXT:    store i16 [[TMP0]], i16* [[__M_ADDR_I]], align 2
// CHECK-NEXT:    store <16 x float> [[TMP1]], <16 x float>* [[__V_ADDR_I]], align 64
// CHECK-NEXT:    store float 0x7FF0000000000000, float* [[__W_ADDR_I_I]], align 4
// CHECK-NEXT:    [[TMP2:%.*]] = load float, float* [[__W_ADDR_I_I]], align 4
// CHECK-NEXT:    [[VECINIT_I_I:%.*]] = insertelement <16 x float> undef, float [[TMP2]], i32 0
// CHECK-NEXT:    [[TMP3:%.*]] = load float, float* [[__W_ADDR_I_I]], align 4
// CHECK-NEXT:    [[VECINIT1_I_I:%.*]] = insertelement <16 x float> [[VECINIT_I_I]], float [[TMP3]], i32 1
// CHECK-NEXT:    [[TMP4:%.*]] = load float, float* [[__W_ADDR_I_I]], align 4
// CHECK-NEXT:    [[VECINIT2_I_I:%.*]] = insertelement <16 x float> [[VECINIT1_I_I]], float [[TMP4]], i32 2
// CHECK-NEXT:    [[TMP5:%.*]] = load float, float* [[__W_ADDR_I_I]], align 4
// CHECK-NEXT:    [[VECINIT3_I_I:%.*]] = insertelement <16 x float> [[VECINIT2_I_I]], float [[TMP5]], i32 3
// CHECK-NEXT:    [[TMP6:%.*]] = load float, float* [[__W_ADDR_I_I]], align 4
// CHECK-NEXT:    [[VECINIT4_I_I:%.*]] = insertelement <16 x float> [[VECINIT3_I_I]], float [[TMP6]], i32 4
// CHECK-NEXT:    [[TMP7:%.*]] = load float, float* [[__W_ADDR_I_I]], align 4
// CHECK-NEXT:    [[VECINIT5_I_I:%.*]] = insertelement <16 x float> [[VECINIT4_I_I]], float [[TMP7]], i32 5
// CHECK-NEXT:    [[TMP8:%.*]] = load float, float* [[__W_ADDR_I_I]], align 4
// CHECK-NEXT:    [[VECINIT6_I_I:%.*]] = insertelement <16 x float> [[VECINIT5_I_I]], float [[TMP8]], i32 6
// CHECK-NEXT:    [[TMP9:%.*]] = load float, float* [[__W_ADDR_I_I]], align 4
// CHECK-NEXT:    [[VECINIT7_I_I:%.*]] = insertelement <16 x float> [[VECINIT6_I_I]], float [[TMP9]], i32 7
// CHECK-NEXT:    [[TMP10:%.*]] = load float, float* [[__W_ADDR_I_I]], align 4
// CHECK-NEXT:    [[VECINIT8_I_I:%.*]] = insertelement <16 x float> [[VECINIT7_I_I]], float [[TMP10]], i32 8
// CHECK-NEXT:    [[TMP11:%.*]] = load float, float* [[__W_ADDR_I_I]], align 4
// CHECK-NEXT:    [[VECINIT9_I_I:%.*]] = insertelement <16 x float> [[VECINIT8_I_I]], float [[TMP11]], i32 9
// CHECK-NEXT:    [[TMP12:%.*]] = load float, float* [[__W_ADDR_I_I]], align 4
// CHECK-NEXT:    [[VECINIT10_I_I:%.*]] = insertelement <16 x float> [[VECINIT9_I_I]], float [[TMP12]], i32 10
// CHECK-NEXT:    [[TMP13:%.*]] = load float, float* [[__W_ADDR_I_I]], align 4
// CHECK-NEXT:    [[VECINIT11_I_I:%.*]] = insertelement <16 x float> [[VECINIT10_I_I]], float [[TMP13]], i32 11
// CHECK-NEXT:    [[TMP14:%.*]] = load float, float* [[__W_ADDR_I_I]], align 4
// CHECK-NEXT:    [[VECINIT12_I_I:%.*]] = insertelement <16 x float> [[VECINIT11_I_I]], float [[TMP14]], i32 12
// CHECK-NEXT:    [[TMP15:%.*]] = load float, float* [[__W_ADDR_I_I]], align 4
// CHECK-NEXT:    [[VECINIT13_I_I:%.*]] = insertelement <16 x float> [[VECINIT12_I_I]], float [[TMP15]], i32 13
// CHECK-NEXT:    [[TMP16:%.*]] = load float, float* [[__W_ADDR_I_I]], align 4
// CHECK-NEXT:    [[VECINIT14_I_I:%.*]] = insertelement <16 x float> [[VECINIT13_I_I]], float [[TMP16]], i32 14
// CHECK-NEXT:    [[TMP17:%.*]] = load float, float* [[__W_ADDR_I_I]], align 4
// CHECK-NEXT:    [[VECINIT15_I_I:%.*]] = insertelement <16 x float> [[VECINIT14_I_I]], float [[TMP17]], i32 15
// CHECK-NEXT:    store <16 x float> [[VECINIT15_I_I]], <16 x float>* [[DOTCOMPOUNDLITERAL_I_I]], align 64
// CHECK-NEXT:    [[TMP18:%.*]] = load <16 x float>, <16 x float>* [[DOTCOMPOUNDLITERAL_I_I]], align 64
// CHECK-NEXT:    [[TMP19:%.*]] = load i16, i16* [[__M_ADDR_I]], align 2
// CHECK-NEXT:    [[TMP20:%.*]] = load <16 x float>, <16 x float>* [[__V_ADDR_I]], align 64
// CHECK-NEXT:    store <16 x float> [[TMP18]], <16 x float>* [[__W2_ADDR_I_I]], align 64
// CHECK-NEXT:    store i16 [[TMP19]], i16* [[__U_ADDR_I_I]], align 2
// CHECK-NEXT:    store <16 x float> [[TMP20]], <16 x float>* [[__A_ADDR_I_I]], align 64
// CHECK-NEXT:    [[TMP21:%.*]] = load i16, i16* [[__U_ADDR_I_I]], align 2
// CHECK-NEXT:    [[TMP22:%.*]] = load <16 x float>, <16 x float>* [[__A_ADDR_I_I]], align 64
// CHECK-NEXT:    [[TMP23:%.*]] = load <16 x float>, <16 x float>* [[__W2_ADDR_I_I]], align 64
// CHECK-NEXT:    [[TMP24:%.*]] = bitcast i16 [[TMP21]] to <16 x i1>
// CHECK-NEXT:    [[TMP25:%.*]] = select <16 x i1> [[TMP24]], <16 x float> [[TMP22]], <16 x float> [[TMP23]]
// CHECK-NEXT:    store <16 x float> [[TMP25]], <16 x float>* [[__V_ADDR_I]], align 64
// CHECK-NEXT:    [[TMP26:%.*]] = load <16 x float>, <16 x float>* [[__V_ADDR_I]], align 64
// CHECK-NEXT:    [[TMP27:%.*]] = bitcast <16 x float> [[TMP26]] to <8 x double>
// CHECK-NEXT:    [[EXTRACT_I:%.*]] = shufflevector <8 x double> [[TMP27]], <8 x double> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// CHECK-NEXT:    [[TMP28:%.*]] = bitcast <4 x double> [[EXTRACT_I]] to <8 x float>
// CHECK-NEXT:    store <8 x float> [[TMP28]], <8 x float>* [[__T1_I]], align 32
// CHECK-NEXT:    [[TMP29:%.*]] = load <16 x float>, <16 x float>* [[__V_ADDR_I]], align 64
// CHECK-NEXT:    [[TMP30:%.*]] = bitcast <16 x float> [[TMP29]] to <8 x double>
// CHECK-NEXT:    [[EXTRACT4_I:%.*]] = shufflevector <8 x double> [[TMP30]], <8 x double> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK-NEXT:    [[TMP31:%.*]] = bitcast <4 x double> [[EXTRACT4_I]] to <8 x float>
// CHECK-NEXT:    store <8 x float> [[TMP31]], <8 x float>* [[__T2_I]], align 32
// CHECK-NEXT:    [[TMP32:%.*]] = load <8 x float>, <8 x float>* [[__T1_I]], align 32
// CHECK-NEXT:    [[TMP33:%.*]] = load <8 x float>, <8 x float>* [[__T2_I]], align 32
// CHECK-NEXT:    store <8 x float> [[TMP32]], <8 x float>* [[__A_ADDR_I16_I]], align 32
// CHECK-NEXT:    store <8 x float> [[TMP33]], <8 x float>* [[__B_ADDR_I17_I]], align 32
// CHECK-NEXT:    [[TMP34:%.*]] = load <8 x float>, <8 x float>* [[__A_ADDR_I16_I]], align 32
// CHECK-NEXT:    [[TMP35:%.*]] = load <8 x float>, <8 x float>* [[__B_ADDR_I17_I]], align 32
// CHECK-NEXT:    [[TMP36:%.*]] = call <8 x float> @llvm.x86.avx.min.ps.256(<8 x float> [[TMP34]], <8 x float> [[TMP35]]) #2
// CHECK-NEXT:    store <8 x float> [[TMP36]], <8 x float>* [[__T3_I]], align 32
// CHECK-NEXT:    [[TMP37:%.*]] = load <8 x float>, <8 x float>* [[__T3_I]], align 32
// CHECK-NEXT:    [[EXTRACT6_I:%.*]] = shufflevector <8 x float> [[TMP37]], <8 x float> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// CHECK-NEXT:    store <4 x float> [[EXTRACT6_I]], <4 x float>* [[__T4_I]], align 16
// CHECK-NEXT:    [[TMP38:%.*]] = load <8 x float>, <8 x float>* [[__T3_I]], align 32
// CHECK-NEXT:    [[EXTRACT7_I:%.*]] = shufflevector <8 x float> [[TMP38]], <8 x float> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK-NEXT:    store <4 x float> [[EXTRACT7_I]], <4 x float>* [[__T5_I]], align 16
// CHECK-NEXT:    [[TMP39:%.*]] = load <4 x float>, <4 x float>* [[__T4_I]], align 16
// CHECK-NEXT:    [[TMP40:%.*]] = load <4 x float>, <4 x float>* [[__T5_I]], align 16
// CHECK-NEXT:    store <4 x float> [[TMP39]], <4 x float>* [[__A_ADDR_I14_I]], align 16
// CHECK-NEXT:    store <4 x float> [[TMP40]], <4 x float>* [[__B_ADDR_I15_I]], align 16
// CHECK-NEXT:    [[TMP41:%.*]] = load <4 x float>, <4 x float>* [[__A_ADDR_I14_I]], align 16
// CHECK-NEXT:    [[TMP42:%.*]] = load <4 x float>, <4 x float>* [[__B_ADDR_I15_I]], align 16
// CHECK-NEXT:    [[TMP43:%.*]] = call <4 x float> @llvm.x86.sse.min.ps(<4 x float> [[TMP41]], <4 x float> [[TMP42]]) #2
// CHECK-NEXT:    store <4 x float> [[TMP43]], <4 x float>* [[__T6_I]], align 16
// CHECK-NEXT:    [[TMP44:%.*]] = load <4 x float>, <4 x float>* [[__T6_I]], align 16
// CHECK-NEXT:    [[TMP45:%.*]] = load <4 x float>, <4 x float>* [[__T6_I]], align 16
// CHECK-NEXT:    [[SHUFFLE_I:%.*]] = shufflevector <4 x float> [[TMP44]], <4 x float> [[TMP45]], <4 x i32> <i32 2, i32 3, i32 0, i32 1>
// CHECK-NEXT:    store <4 x float> [[SHUFFLE_I]], <4 x float>* [[__T7_I]], align 16
// CHECK-NEXT:    [[TMP46:%.*]] = load <4 x float>, <4 x float>* [[__T6_I]], align 16
// CHECK-NEXT:    [[TMP47:%.*]] = load <4 x float>, <4 x float>* [[__T7_I]], align 16
// CHECK-NEXT:    store <4 x float> [[TMP46]], <4 x float>* [[__A_ADDR_I12_I]], align 16
// CHECK-NEXT:    store <4 x float> [[TMP47]], <4 x float>* [[__B_ADDR_I13_I]], align 16
// CHECK-NEXT:    [[TMP48:%.*]] = load <4 x float>, <4 x float>* [[__A_ADDR_I12_I]], align 16
// CHECK-NEXT:    [[TMP49:%.*]] = load <4 x float>, <4 x float>* [[__B_ADDR_I13_I]], align 16
// CHECK-NEXT:    [[TMP50:%.*]] = call <4 x float> @llvm.x86.sse.min.ps(<4 x float> [[TMP48]], <4 x float> [[TMP49]]) #2
// CHECK-NEXT:    store <4 x float> [[TMP50]], <4 x float>* [[__T8_I]], align 16
// CHECK-NEXT:    [[TMP51:%.*]] = load <4 x float>, <4 x float>* [[__T8_I]], align 16
// CHECK-NEXT:    [[TMP52:%.*]] = load <4 x float>, <4 x float>* [[__T8_I]], align 16
// CHECK-NEXT:    [[SHUFFLE10_I:%.*]] = shufflevector <4 x float> [[TMP51]], <4 x float> [[TMP52]], <4 x i32> <i32 1, i32 0, i32 3, i32 2>
// CHECK-NEXT:    store <4 x float> [[SHUFFLE10_I]], <4 x float>* [[__T9_I]], align 16
// CHECK-NEXT:    [[TMP53:%.*]] = load <4 x float>, <4 x float>* [[__T8_I]], align 16
// CHECK-NEXT:    [[TMP54:%.*]] = load <4 x float>, <4 x float>* [[__T9_I]], align 16
// CHECK-NEXT:    store <4 x float> [[TMP53]], <4 x float>* [[__A2_ADDR_I_I]], align 16
// CHECK-NEXT:    store <4 x float> [[TMP54]], <4 x float>* [[__B_ADDR_I_I]], align 16
// CHECK-NEXT:    [[TMP55:%.*]] = load <4 x float>, <4 x float>* [[__A2_ADDR_I_I]], align 16
// CHECK-NEXT:    [[TMP56:%.*]] = load <4 x float>, <4 x float>* [[__B_ADDR_I_I]], align 16
// CHECK-NEXT:    [[TMP57:%.*]] = call <4 x float> @llvm.x86.sse.min.ps(<4 x float> [[TMP55]], <4 x float> [[TMP56]]) #2
// CHECK-NEXT:    store <4 x float> [[TMP57]], <4 x float>* [[__T10_I]], align 16
// CHECK-NEXT:    [[TMP58:%.*]] = load <4 x float>, <4 x float>* [[__T10_I]], align 16
// CHECK-NEXT:    [[VECEXT_I:%.*]] = extractelement <4 x float> [[TMP58]], i32 0
// CHECK-NEXT:    ret float [[VECEXT_I]]
float test_mm512_mask_reduce_min_ps(__mmask16 __M, __m512 __W){
  return _mm512_mask_reduce_min_ps(__M, __W); 
}

