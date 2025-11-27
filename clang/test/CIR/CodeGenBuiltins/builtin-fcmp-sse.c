// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

typedef float __m128 __attribute__((__vector_size__(16), __aligned__(16)));
typedef double __m128d __attribute__((__vector_size__(16), __aligned__(16)));

__m128 test_cmpnleps(__m128 A, __m128 B) {
  // CIR-LABEL:   cir.func dso_local @test_cmpnleps(
  // CIR:           %[[ARG0:.*]]: !cir.vector<4 x !cir.float> {{.*}}, %[[ARG1:.*]]: !cir.vector<4 x !cir.float> {{.*}}) -> !cir.vector<4 x !cir.float> inline(never) {
  // CIR:           %[[ALLOCA_0:.*]] = cir.alloca !cir.vector<4 x !cir.float>, !cir.ptr<!cir.vector<4 x !cir.float>>, ["A", init] {alignment = 16 : i64}
  // CIR:           %[[ALLOCA_1:.*]] = cir.alloca !cir.vector<4 x !cir.float>, !cir.ptr<!cir.vector<4 x !cir.float>>, ["B", init] {alignment = 16 : i64}
  // CIR:           %[[ALLOCA_2:.*]] = cir.alloca !cir.vector<4 x !cir.float>, !cir.ptr<!cir.vector<4 x !cir.float>>, ["__retval"] {alignment = 16 : i64}
  // CIR:           cir.store %[[ARG0]], %[[ALLOCA_0]] : !cir.vector<4 x !cir.float>, !cir.ptr<!cir.vector<4 x !cir.float>>
  // CIR:           cir.store %[[ARG1]], %[[ALLOCA_1]] : !cir.vector<4 x !cir.float>, !cir.ptr<!cir.vector<4 x !cir.float>>
  // CIR:           %[[LOAD_0:.*]] = cir.load align(16) %[[ALLOCA_0]] : !cir.ptr<!cir.vector<4 x !cir.float>>, !cir.vector<4 x !cir.float>
  // CIR:           %[[LOAD_1:.*]] = cir.load align(16) %[[ALLOCA_1]] : !cir.ptr<!cir.vector<4 x !cir.float>>, !cir.vector<4 x !cir.float>
  // CIR:           %[[VEC_0:.*]] = cir.vec.cmp(le, %[[LOAD_0]], %[[LOAD_1]]) : !cir.vector<4 x !cir.float>, !cir.vector<4 x !s32i>
  // CIR:           %[[UNARY_0:.*]] = cir.unary(not, %[[VEC_0]]) : !cir.vector<4 x !s32i>, !cir.vector<4 x !s32i>
  // CIR:           %[[CAST_0:.*]] = cir.cast bitcast %[[UNARY_0]] : !cir.vector<4 x !s32i> -> !cir.vector<4 x !cir.float>
  // CIR:           cir.store %[[CAST_0]], %[[ALLOCA_2]] : !cir.vector<4 x !cir.float>, !cir.ptr<!cir.vector<4 x !cir.float>>
  // CIR:           %[[LOAD_2:.*]] = cir.load %[[ALLOCA_2]] : !cir.ptr<!cir.vector<4 x !cir.float>>, !cir.vector<4 x !cir.float>
  // CIR:           cir.return %[[LOAD_2]] : !cir.vector<4 x !cir.float>
  // CIR:         } 

  // LLVM-LABEL: define dso_local <4 x float> @test_cmpnleps(
  // LLVM-SAME: <4 x float> [[TMP0:%.*]], <4 x float> [[TMP1:%.*]]) #[[ATTR0:[0-9]+]] {
  // LLVM-NEXT:    [[TMP3:%.*]] = alloca <4 x float>, i64 1, align 16
  // LLVM-NEXT:    [[TMP4:%.*]] = alloca <4 x float>, i64 1, align 16
  // LLVM-NEXT:    [[TMP5:%.*]] = alloca <4 x float>, i64 1, align 16
  // LLVM-NEXT:    store <4 x float> [[TMP0]], ptr [[TMP3]], align 16
  // LLVM-NEXT:    store <4 x float> [[TMP1]], ptr [[TMP4]], align 16
  // LLVM-NEXT:    [[TMP6:%.*]] = load <4 x float>, ptr [[TMP3]], align 16
  // LLVM-NEXT:    [[TMP7:%.*]] = load <4 x float>, ptr [[TMP4]], align 16
  // LLVM-NEXT:    [[TMP8:%.*]] = fcmp ole <4 x float> [[TMP6]], [[TMP7]]
  // LLVM-NEXT:    [[TMP9:%.*]] = sext <4 x i1> [[TMP8]] to <4 x i32>
  // LLVM-NEXT:    [[TMP10:%.*]] = xor <4 x i32> [[TMP9]], splat (i32 -1)
  // LLVM-NEXT:    [[TMP11:%.*]] = bitcast <4 x i32> [[TMP10]] to <4 x float>
  // LLVM-NEXT:    store <4 x float> [[TMP11]], ptr [[TMP5]], align 16
  // LLVM-NEXT:    [[TMP12:%.*]] = load <4 x float>, ptr [[TMP5]], align 16
  // LLVM-NEXT:    ret <4 x float> [[TMP12]]

  // OGCG-LABEL: define dso_local <4 x float> @test_cmpnleps(
  // OGCG-SAME: <4 x float> noundef [[A:%.*]], <4 x float> noundef [[B:%.*]]) #[[ATTR0:[0-9]+]] {
  // OGCG-NEXT:  [[ENTRY:.*:]]
  // OGCG-NEXT:    [[A_ADDR:%.*]] = alloca <4 x float>, align 16
  // OGCG-NEXT:    [[B_ADDR:%.*]] = alloca <4 x float>, align 16
  // OGCG-NEXT:    store <4 x float> [[A]], ptr [[A_ADDR]], align 16
  // OGCG-NEXT:    store <4 x float> [[B]], ptr [[B_ADDR]], align 16
  // OGCG-NEXT:    [[TMP0:%.*]] = load <4 x float>, ptr [[A_ADDR]], align 16
  // OGCG-NEXT:    [[TMP1:%.*]] = load <4 x float>, ptr [[B_ADDR]], align 16
  // OGCG-NEXT:    [[TMP2:%.*]] = fcmp ugt <4 x float> [[TMP0]], [[TMP1]]
  // OGCG-NEXT:    [[TMP3:%.*]] = sext <4 x i1> [[TMP2]] to <4 x i32>
  // OGCG-NEXT:    [[TMP4:%.*]] = bitcast <4 x i32> [[TMP3]] to <4 x float>
  // OGCG-NEXT:    ret <4 x float> [[TMP4]]
  return __builtin_ia32_cmpnleps(A, B);
}

__m128d test_cmpnlepd(__m128d A, __m128d B) {
  // CIR-LABEL:   cir.func dso_local @test_cmpnlepd(
  // CIR：          %[[ARG0:.*]]: !cir.vector<2 x !cir.double> {{.*}}, %[[ARG1:.*]]: !cir.vector<2 x !cir.double> {{.*}}) -> !cir.vector<2 x !cir.double> inline(never) {
  // CIR:           %[[ALLOCA_0:.*]] = cir.alloca !cir.vector<2 x !cir.double>, !cir.ptr<!cir.vector<2 x !cir.double>>, ["A", init] {alignment = 16 : i64} 
  // CIR:           %[[ALLOCA_1:.*]] = cir.alloca !cir.vector<2 x !cir.double>, !cir.ptr<!cir.vector<2 x !cir.double>>, ["B", init] {alignment = 16 : i64} 
  // CIR:           %[[ALLOCA_2:.*]] = cir.alloca !cir.vector<2 x !cir.double>, !cir.ptr<!cir.vector<2 x !cir.double>>, ["__retval"] {alignment = 16 : i64} 
  // CIR:           cir.store %[[ARG0]], %[[ALLOCA_0]] : !cir.vector<2 x !cir.double>, !cir.ptr<!cir.vector<2 x !cir.double>> 
  // CIR:           cir.store %[[ARG1]], %[[ALLOCA_1]] : !cir.vector<2 x !cir.double>, !cir.ptr<!cir.vector<2 x !cir.double>> 
  // CIR:           %[[LOAD_0:.*]] = cir.load align(16) %[[ALLOCA_0]] : !cir.ptr<!cir.vector<2 x !cir.double>>, !cir.vector<2 x !cir.double> 
  // CIR:           %[[LOAD_1:.*]] = cir.load align(16) %[[ALLOCA_1]] : !cir.ptr<!cir.vector<2 x !cir.double>>, !cir.vector<2 x !cir.double> 
  // CIR:           %[[VEC_0:.*]] = cir.vec.cmp(le, %[[LOAD_0]], %[[LOAD_1]]) : !cir.vector<2 x !cir.double>, !cir.vector<2 x !s64i> 
  // CIR:           %[[UNARY_0:.*]] = cir.unary(not, %[[VEC_0]]) : !cir.vector<2 x !s64i>, !cir.vector<2 x !s64i> 
  // CIR:           %[[CAST_0:.*]] = cir.cast bitcast %[[UNARY_0]] : !cir.vector<2 x !s64i> -> !cir.vector<2 x !cir.double> 
  // CIR:           cir.store %[[CAST_0]], %[[ALLOCA_2]] : !cir.vector<2 x !cir.double>, !cir.ptr<!cir.vector<2 x !cir.double>> 
  // CIR:           %[[LOAD_2:.*]] = cir.load %[[ALLOCA_2]] : !cir.ptr<!cir.vector<2 x !cir.double>>, !cir.vector<2 x !cir.double> 
  // CIR:           cir.return %[[LOAD_2]] : !cir.vector<2 x !cir.double> 
  // CIR:         } 

  // LLVM-LABEL: define dso_local <2 x double> @test_cmpnlepd(
  // LLVM-SAME: <2 x double> [[TMP0:%.*]], <2 x double> [[TMP1:%.*]]) #[[ATTR0:[0-9]+]] {
  // LLVM-NEXT:    [[TMP3:%.*]] = alloca <2 x double>, i64 1, align 16
  // LLVM-NEXT:    [[TMP4:%.*]] = alloca <2 x double>, i64 1, align 16
  // LLVM-NEXT:    [[TMP5:%.*]] = alloca <2 x double>, i64 1, align 16
  // LLVM-NEXT:    store <2 x double> [[TMP0]], ptr [[TMP3]], align 16
  // LLVM-NEXT:    store <2 x double> [[TMP1]], ptr [[TMP4]], align 16
  // LLVM-NEXT:    [[TMP6:%.*]] = load <2 x double>, ptr [[TMP3]], align 16
  // LLVM-NEXT:    [[TMP7:%.*]] = load <2 x double>, ptr [[TMP4]], align 16
  // LLVM-NEXT:    [[TMP8:%.*]] = fcmp ole <2 x double> [[TMP6]], [[TMP7]]
  // LLVM-NEXT:    [[TMP9:%.*]] = sext <2 x i1> [[TMP8]] to <2 x i64>
  // LLVM-NEXT:    [[TMP10:%.*]] = xor <2 x i64> [[TMP9]], splat (i64 -1)
  // LLVM-NEXT:    [[TMP11:%.*]] = bitcast <2 x i64> [[TMP10]] to <2 x double>
  // LLVM-NEXT:    store <2 x double> [[TMP11]], ptr [[TMP5]], align 16
  // LLVM-NEXT:    [[TMP12:%.*]] = load <2 x double>, ptr [[TMP5]], align 16
  // LLVM-NEXT:    ret <2 x double> [[TMP12]]

  // OGCG-LABEL: define dso_local <2 x double> @test_cmpnlepd(
  // OGCG-SAME: <2 x double> noundef [[A:%.*]], <2 x double> noundef [[B:%.*]]) #[[ATTR0:[0-9]+]] {
  // OGCG-NEXT:  [[ENTRY:.*:]]
  // OGCG-NEXT:    [[A_ADDR:%.*]] = alloca <2 x double>, align 16
  // OGCG-NEXT:    [[B_ADDR:%.*]] = alloca <2 x double>, align 16
  // OGCG-NEXT:    store <2 x double> [[A]], ptr [[A_ADDR]], align 16
  // OGCG-NEXT:    store <2 x double> [[B]], ptr [[B_ADDR]], align 16
  // OGCG-NEXT:    [[TMP0:%.*]] = load <2 x double>, ptr [[A_ADDR]], align 16
  // OGCG-NEXT:    [[TMP1:%.*]] = load <2 x double>, ptr [[B_ADDR]], align 16
  // OGCG-NEXT:    [[TMP2:%.*]] = fcmp ugt <2 x double> [[TMP0]], [[TMP1]]
  // OGCG-NEXT:    [[TMP3:%.*]] = sext <2 x i1> [[TMP2]] to <2 x i64>
  // OGCG-NEXT:    [[TMP4:%.*]] = bitcast <2 x i64> [[TMP3]] to <2 x double>
  // OGCG-NEXT:    ret <2 x double> [[TMP4]]
 return  __builtin_ia32_cmpnlepd(A, B);
}

__m128 test_cmpnltps(__m128 A, __m128 B) {
  // CIR-LABEL:   cir.func dso_local @test_cmpnltps(
  // CIR-SAME:      %[[ARG0:.*]]: !cir.vector<4 x !cir.float> {{.*}}, %[[ARG1:.*]]: !cir.vector<4 x !cir.float> {{.*}}) -> !cir.vector<4 x !cir.float> inline(never) {
  // CIR:           %[[ALLOCA_0:.*]] = cir.alloca !cir.vector<4 x !cir.float>, !cir.ptr<!cir.vector<4 x !cir.float>>, ["A", init] {alignment = 16 : i64} 
  // CIR:           %[[ALLOCA_1:.*]] = cir.alloca !cir.vector<4 x !cir.float>, !cir.ptr<!cir.vector<4 x !cir.float>>, ["B", init] {alignment = 16 : i64} 
  // CIR:           %[[ALLOCA_2:.*]] = cir.alloca !cir.vector<4 x !cir.float>, !cir.ptr<!cir.vector<4 x !cir.float>>, ["__retval"] {alignment = 16 : i64} 
  // CIR:           cir.store %[[ARG0]], %[[ALLOCA_0]] : !cir.vector<4 x !cir.float>, !cir.ptr<!cir.vector<4 x !cir.float>> 
  // CIR:           cir.store %[[ARG1]], %[[ALLOCA_1]] : !cir.vector<4 x !cir.float>, !cir.ptr<!cir.vector<4 x !cir.float>> 
  // CIR:           %[[LOAD_0:.*]] = cir.load align(16) %[[ALLOCA_0]] : !cir.ptr<!cir.vector<4 x !cir.float>>, !cir.vector<4 x !cir.float> 
  // CIR:           %[[LOAD_1:.*]] = cir.load align(16) %[[ALLOCA_1]] : !cir.ptr<!cir.vector<4 x !cir.float>>, !cir.vector<4 x !cir.float> 
  // CIR:           %[[VEC_0:.*]] = cir.vec.cmp(lt, %[[LOAD_0]], %[[LOAD_1]]) : !cir.vector<4 x !cir.float>, !cir.vector<4 x !s32i> 
  // CIR:           %[[UNARY_0:.*]] = cir.unary(not, %[[VEC_0]]) : !cir.vector<4 x !s32i>, !cir.vector<4 x !s32i> 
  // CIR:           %[[CAST_0:.*]] = cir.cast bitcast %[[UNARY_0]] : !cir.vector<4 x !s32i> -> !cir.vector<4 x !cir.float> 
  // CIR:           cir.store %[[CAST_0]], %[[ALLOCA_2]] : !cir.vector<4 x !cir.float>, !cir.ptr<!cir.vector<4 x !cir.float>> 
  // CIR:           %[[LOAD_2:.*]] = cir.load %[[ALLOCA_2]] : !cir.ptr<!cir.vector<4 x !cir.float>>, !cir.vector<4 x !cir.float> 
  // CIR:           cir.return %[[LOAD_2]] : !cir.vector<4 x !cir.float> 
  // CIR:         } 

  // LLVM-LABEL: define dso_local <4 x float> @test_cmpnltps(
  // LLVM-SAME: <4 x float> [[TMP0:%.*]], <4 x float> [[TMP1:%.*]]) #[[ATTR0:[0-9]+]] {
  // LLVM-NEXT:    [[TMP3:%.*]] = alloca <4 x float>, i64 1, align 16
  // LLVM-NEXT:    [[TMP4:%.*]] = alloca <4 x float>, i64 1, align 16
  // LLVM-NEXT:    [[TMP5:%.*]] = alloca <4 x float>, i64 1, align 16
  // LLVM-NEXT:    store <4 x float> [[TMP0]], ptr [[TMP3]], align 16
  // LLVM-NEXT:    store <4 x float> [[TMP1]], ptr [[TMP4]], align 16
  // LLVM-NEXT:    [[TMP6:%.*]] = load <4 x float>, ptr [[TMP3]], align 16
  // LLVM-NEXT:    [[TMP7:%.*]] = load <4 x float>, ptr [[TMP4]], align 16
  // LLVM-NEXT:    [[TMP8:%.*]] = fcmp olt <4 x float> [[TMP6]], [[TMP7]]
  // LLVM-NEXT:    [[TMP9:%.*]] = sext <4 x i1> [[TMP8]] to <4 x i32>
  // LLVM-NEXT:    [[TMP10:%.*]] = xor <4 x i32> [[TMP9]], splat (i32 -1)
  // LLVM-NEXT:    [[TMP11:%.*]] = bitcast <4 x i32> [[TMP10]] to <4 x float>
  // LLVM-NEXT:    store <4 x float> [[TMP11]], ptr [[TMP5]], align 16
  // LLVM-NEXT:    [[TMP12:%.*]] = load <4 x float>, ptr [[TMP5]], align 16
  // LLVM-NEXT:    ret <4 x float> [[TMP12]]

  // OGCG-LABEL: define dso_local <4 x float> @test_cmpnltps(
  // OGCG-SAME: <4 x float> noundef [[A:%.*]], <4 x float> noundef [[B:%.*]]) #[[ATTR0:[0-9]+]] {
  // OGCG-NEXT:  [[ENTRY:.*:]]
  // OGCG-NEXT:    [[A_ADDR:%.*]] = alloca <4 x float>, align 16
  // OGCG-NEXT:    [[B_ADDR:%.*]] = alloca <4 x float>, align 16
  // OGCG-NEXT:    store <4 x float> [[A]], ptr [[A_ADDR]], align 16
  // OGCG-NEXT:    store <4 x float> [[B]], ptr [[B_ADDR]], align 16
  // OGCG-NEXT:    [[TMP0:%.*]] = load <4 x float>, ptr [[A_ADDR]], align 16
  // OGCG-NEXT:    [[TMP1:%.*]] = load <4 x float>, ptr [[B_ADDR]], align 16
  // OGCG-NEXT:    [[TMP2:%.*]] = fcmp uge <4 x float> [[TMP0]], [[TMP1]]
  // OGCG-NEXT:    [[TMP3:%.*]] = sext <4 x i1> [[TMP2]] to <4 x i32>
  // OGCG-NEXT:    [[TMP4:%.*]] = bitcast <4 x i32> [[TMP3]] to <4 x float>
  // OGCG-NEXT:    ret <4 x float> [[TMP4]]
  return __builtin_ia32_cmpnltps(A, B);
}

__m128d test_cmpnltpd(__m128d A, __m128d B) {
  // CIR-LABEL:   cir.func dso_local @test_cmpnltpd(
  // CIR:           %[[ARG0:.*]]: !cir.vector<2 x !cir.double> {{.*}}, %[[ARG1:.*]]: !cir.vector<2 x !cir.double> {{.*}}) -> !cir.vector<2 x !cir.double> inline(never) {
  // CIR:           %[[ALLOCA_0:.*]] = cir.alloca !cir.vector<2 x !cir.double>, !cir.ptr<!cir.vector<2 x !cir.double>>, ["A", init] {alignment = 16 : i64} 
  // CIR:           %[[ALLOCA_1:.*]] = cir.alloca !cir.vector<2 x !cir.double>, !cir.ptr<!cir.vector<2 x !cir.double>>, ["B", init] {alignment = 16 : i64} 
  // CIR:           %[[ALLOCA_2:.*]] = cir.alloca !cir.vector<2 x !cir.double>, !cir.ptr<!cir.vector<2 x !cir.double>>, ["__retval"] {alignment = 16 : i64} 
  // CIR:           cir.store %[[ARG0]], %[[ALLOCA_0]] : !cir.vector<2 x !cir.double>, !cir.ptr<!cir.vector<2 x !cir.double>> 
  // CIR:           cir.store %[[ARG1]], %[[ALLOCA_1]] : !cir.vector<2 x !cir.double>, !cir.ptr<!cir.vector<2 x !cir.double>> 
  // CIR:           %[[LOAD_0:.*]] = cir.load align(16) %[[ALLOCA_0]] : !cir.ptr<!cir.vector<2 x !cir.double>>, !cir.vector<2 x !cir.double> 
  // CIR:           %[[LOAD_1:.*]] = cir.load align(16) %[[ALLOCA_1]] : !cir.ptr<!cir.vector<2 x !cir.double>>, !cir.vector<2 x !cir.double> 
  // CIR:           %[[VEC_0:.*]] = cir.vec.cmp(lt, %[[LOAD_0]], %[[LOAD_1]]) : !cir.vector<2 x !cir.double>, !cir.vector<2 x !s64i> 
  // CIR:           %[[UNARY_0:.*]] = cir.unary(not, %[[VEC_0]]) : !cir.vector<2 x !s64i>, !cir.vector<2 x !s64i> 
  // CIR:           %[[CAST_0:.*]] = cir.cast bitcast %[[UNARY_0]] : !cir.vector<2 x !s64i> -> !cir.vector<2 x !cir.double> 
  // CIR:           cir.store %[[CAST_0]], %[[ALLOCA_2]] : !cir.vector<2 x !cir.double>, !cir.ptr<!cir.vector<2 x !cir.double>> 
  // CIR:           %[[LOAD_2:.*]] = cir.load %[[ALLOCA_2]] : !cir.ptr<!cir.vector<2 x !cir.double>>, !cir.vector<2 x !cir.double> 
  // CIR:           cir.return %[[LOAD_2]] : !cir.vector<2 x !cir.double> 
  // CIR:         } 

  // LLVM-LABEL: define dso_local <2 x double> @test_cmpnltpd(
  // LLVM-SAME: <2 x double> [[TMP0:%.*]], <2 x double> [[TMP1:%.*]]) #[[ATTR0:[0-9]+]] {
  // LLVM-NEXT:    [[TMP3:%.*]] = alloca <2 x double>, i64 1, align 16
  // LLVM-NEXT:    [[TMP4:%.*]] = alloca <2 x double>, i64 1, align 16
  // LLVM-NEXT:    [[TMP5:%.*]] = alloca <2 x double>, i64 1, align 16
  // LLVM-NEXT:    store <2 x double> [[TMP0]], ptr [[TMP3]], align 16
  // LLVM-NEXT:    store <2 x double> [[TMP1]], ptr [[TMP4]], align 16
  // LLVM-NEXT:    [[TMP6:%.*]] = load <2 x double>, ptr [[TMP3]], align 16
  // LLVM-NEXT:    [[TMP7:%.*]] = load <2 x double>, ptr [[TMP4]], align 16
  // LLVM-NEXT:    [[TMP8:%.*]] = fcmp olt <2 x double> [[TMP6]], [[TMP7]]
  // LLVM-NEXT:    [[TMP9:%.*]] = sext <2 x i1> [[TMP8]] to <2 x i64>
  // LLVM-NEXT:    [[TMP10:%.*]] = xor <2 x i64> [[TMP9]], splat (i64 -1)
  // LLVM-NEXT:    [[TMP11:%.*]] = bitcast <2 x i64> [[TMP10]] to <2 x double>
  // LLVM-NEXT:    store <2 x double> [[TMP11]], ptr [[TMP5]], align 16
  // LLVM-NEXT:    [[TMP12:%.*]] = load <2 x double>, ptr [[TMP5]], align 16
  // LLVM-NEXT:    ret <2 x double> [[TMP12]]

  // OGCG-LABEL: define dso_local <2 x double> @test_cmpnltpd(
  // OGCG-SAME: <2 x double> noundef [[A:%.*]], <2 x double> noundef [[B:%.*]]) #[[ATTR0:[0-9]+]] {
  // OGCG-NEXT:  [[ENTRY:.*:]]
  // OGCG-NEXT:    [[A_ADDR:%.*]] = alloca <2 x double>, align 16
  // OGCG-NEXT:    [[B_ADDR:%.*]] = alloca <2 x double>, align 16
  // OGCG-NEXT:    store <2 x double> [[A]], ptr [[A_ADDR]], align 16
  // OGCG-NEXT:    store <2 x double> [[B]], ptr [[B_ADDR]], align 16
  // OGCG-NEXT:    [[TMP0:%.*]] = load <2 x double>, ptr [[A_ADDR]], align 16
  // OGCG-NEXT:    [[TMP1:%.*]] = load <2 x double>, ptr [[B_ADDR]], align 16
  // OGCG-NEXT:    [[TMP2:%.*]] = fcmp uge <2 x double> [[TMP0]], [[TMP1]]
  // OGCG-NEXT:    [[TMP3:%.*]] = sext <2 x i1> [[TMP2]] to <2 x i64>
  // OGCG-NEXT:    [[TMP4:%.*]] = bitcast <2 x i64> [[TMP3]] to <2 x double>
  // OGCG-NEXT:    ret <2 x double> [[TMP4]]
  return  __builtin_ia32_cmpnltpd(A, B);
}
