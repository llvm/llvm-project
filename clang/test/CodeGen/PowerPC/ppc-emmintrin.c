// REQUIRES: powerpc-registered-target

// RUN: %clang -S -emit-llvm -target powerpc64-unknown-linux-gnu -mcpu=pwr8 -ffreestanding -DNO_WARN_X86_INTRINSICS %s \
// RUN:  -ffp-contract=off -fno-discard-value-names -mllvm -disable-llvm-optzns -o - | llvm-cxxfilt -n | FileCheck %s --check-prefixes=CHECK,CHECK-BE
// RUN: %clang -S -emit-llvm -target powerpc64le-unknown-linux-gnu -mcpu=pwr8 -ffreestanding -DNO_WARN_X86_INTRINSICS %s \
// RUN:   -ffp-contract=off -fno-discard-value-names -mllvm -disable-llvm-optzns -o - | llvm-cxxfilt -n | FileCheck %s --check-prefixes=CHECK,CHECK-LE

// RUN: %clang -S -emit-llvm -target powerpc64le-unknown-linux-gnu -mcpu=pwr10 -ffreestanding -DNO_WARN_X86_INTRINSICS %s \
// RUN:   -ffp-contract=off -fno-discard-value-names -mllvm -disable-llvm-optzns -o - | llvm-cxxfilt -n | FileCheck %s --check-prefixes=CHECK-P10-LE

// CHECK-BE-DAG: @_mm_movemask_pd.__perm_mask = internal constant <4 x i32> <i32 -2139062144, i32 -2139062144, i32 -2139062144, i32 -2139078656>, align 16
// CHECK-BE-DAG: @_mm_shuffle_epi32.__permute_selectors = internal constant [4 x i32] [i32 66051, i32 67438087, i32 134810123, i32 202182159], align 4
// CHECK-BE-DAG: @_mm_shufflehi_epi16.__permute_selectors = internal constant [4 x i16] [i16 2057, i16 2571, i16 3085, i16 3599], align 2
// CHECK-BE-DAG: @_mm_shufflelo_epi16.__permute_selectors = internal constant [4 x i16] [i16 1, i16 515, i16 1029, i16 1543], align 2

// CHECK-LE-DAG: @_mm_movemask_pd.__perm_mask = internal constant <4 x i32> <i32 -2139094976, i32 -2139062144, i32 -2139062144, i32 -2139062144>, align 16
// CHECK-LE-DAG: @_mm_shuffle_epi32.__permute_selectors = internal constant [4 x i32] [i32 50462976, i32 117835012, i32 185207048, i32 252579084], align 4
// CHECK-LE-DAG: @_mm_shufflehi_epi16.__permute_selectors = internal constant [4 x i16] [i16 2312, i16 2826, i16 3340, i16 3854], align 2
// CHECK-LE-DAG: @_mm_shufflelo_epi16.__permute_selectors = internal constant [4 x i16] [i16 256, i16 770, i16 1284, i16 1798], align 2

#include <emmintrin.h>

__m128i resi, mi1, mi2;
__m128i *mip;
double dp[2];
__m128d resd, md1, md2;
__m64 res64, m641, m642;
__m128 res, m1;
int i;
char chs[16];
int is[4];
short ss[8];
long long i64s[2];

void __attribute__((noinline))
test_add() {
  resi = _mm_add_epi64(mi1, mi2);
  resi = _mm_add_epi32(mi1, mi2);
  resi = _mm_add_epi16(mi1, mi2);
  resi = _mm_add_epi8(mi1, mi2);
  resd = _mm_add_pd(md1, md2);
  resd = _mm_add_sd(md1, md2);
  res64 = _mm_add_si64(m641, m642);
  resi = _mm_adds_epi16(mi1, mi2);
  resi = _mm_adds_epi8(mi1, mi2);
  resi = _mm_adds_epu16(mi1, mi2);
  resi = _mm_adds_epu8(mi1, mi2);
}

// CHECK-LABEL: @test_add

// CHECK-LABEL: define available_externally <2 x i64> @_mm_add_epi64
// CHECK: add <2 x i64>

// CHECK-LABEL: define available_externally <2 x i64> @_mm_add_epi32
// CHECK: add <4 x i32>

// CHECK-LABEL: define available_externally <2 x i64> @_mm_add_epi16
// CHECK: add <8 x i16>

// CHECK-LABEL: define available_externally <2 x i64> @_mm_add_epi8
// CHECK: add <16 x i8>

// CHECK-LABEL: define available_externally <2 x double> @_mm_add_pd
// CHECK: fadd <2 x double>

// CHECK-LABEL: define available_externally <2 x double> @_mm_add_sd
// CHECK: fadd double

// CHECK-LABEL: define available_externally i64 @_mm_add_si64
// CHECK: add i64

// CHECK-LABEL: define available_externally <2 x i64> @_mm_adds_epi16
// CHECK: call <8 x i16> @vec_adds(short vector[8], short vector[8])

// CHECK-LABEL: define available_externally <2 x i64> @_mm_adds_epi8
// CHECK: call <16 x i8> @vec_adds(signed char vector[16], signed char vector[16])

// CHECK-LABEL: define available_externally <2 x i64> @_mm_adds_epu16
// CHECK: call <8 x i16> @vec_adds(unsigned short vector[8], unsigned short vector[8])

// CHECK-LABEL: define available_externally <2 x i64> @_mm_adds_epu8
// CHECK: call <16 x i8> @vec_adds(unsigned char vector[16], unsigned char vector[16])

void __attribute__((noinline))
test_avg() {
  resi = _mm_avg_epu16(mi1, mi2);
  resi = _mm_avg_epu8(mi1, mi2);
}

// CHECK-LABEL: @test_avg

// CHECK-LABEL: define available_externally <2 x i64> @_mm_avg_epu16
// CHECK: call <8 x i16> @vec_avg(unsigned short vector[8], unsigned short vector[8])

// CHECK-LABEL: define available_externally <2 x i64> @_mm_avg_epu8
// CHECK: call <16 x i8> @vec_avg(unsigned char vector[16], unsigned char vector[16])

void __attribute__((noinline))
test_bs() {
  resi = _mm_bslli_si128(mi1, i);
  resi = _mm_bsrli_si128(mi1, i);
}

// CHECK-LABEL: @test_bs

// CHECK-LABEL: define available_externally <2 x i64> @_mm_bslli_si128
// CHECK: %[[CMP:[0-9a-zA-Z_.]+]] = icmp slt i32 %{{[0-9a-zA-Z_.]+}}, 16
// CHECK: br i1 %[[CMP]]
// CHECK: call <16 x i8> @vec_sld(unsigned char vector[16], unsigned char vector[16], unsigned int)(<16 x i8> noundef %{{[0-9a-zA-Z_.]+}}, <16 x i8> noundef zeroinitializer, i32 noundef zeroext %{{[0-9a-zA-Z_.]+}})
// CHECK: store <16 x i8> zeroinitializer, <16 x i8>* %{{[0-9a-zA-Z_.]+}}, align 16

// CHECK-LABEL: define available_externally <2 x i64> @_mm_bsrli_si128
// CHECK: %[[CMP:[0-9a-zA-Z_.]+]] = icmp slt i32 %{{[0-9a-zA-Z_.]+}}, 16
// CHECK: br i1 %[[CMP]]
// CHECK-LE: call i1 @llvm.is.constant
// CHECK-LE: %[[SUB:[0-9a-zA-Z_.]+]] = sub nsw i32 16, %{{[0-9a-zA-Z_.]+}}
// CHECK-LE: call <16 x i8> @vec_sld(unsigned char vector[16], unsigned char vector[16], unsigned int)(<16 x i8> noundef zeroinitializer, <16 x i8> noundef %{{[0-9a-zA-Z_.]+}}, i32 noundef zeroext %[[SUB]])
// CHECK-LE: %[[MUL:[0-9a-zA-Z_.]+]] = mul nsw i32 %{{[0-9a-zA-Z_.]+}}, 8
// CHECK-LE: %[[TRUNC:[0-9a-zA-Z_.]+]] = trunc i32 %[[MUL]] to i8
// CHECK-LE: call <16 x i8> @vec_splats(unsigned char)(i8 noundef zeroext %[[TRUNC]])
// CHECK-LE: call <16 x i8> @vec_sro(unsigned char vector[16], unsigned char vector[16])
// CHECK-LE: store <16 x i8> zeroinitializer, <16 x i8>* %{{[0-9a-zA-Z_.]+}}, align 16
// CHECK-BE: %[[MUL:[0-9a-zA-Z_.]+]] = mul nsw i32 %{{[0-9a-zA-Z_.]+}}, 8
// CHECK-BE: %[[TRUNC:[0-9a-zA-Z_.]+]] = trunc i32 %[[MUL]] to i8
// CHECK-BE: call <16 x i8> @vec_splats(unsigned char)(i8 noundef zeroext %[[TRUNC]])
// CHECK-BE: call <16 x i8> @vec_slo(unsigned char vector[16], unsigned char vector[16])
// CHECK-BE: store <16 x i8> zeroinitializer, <16 x i8>* %{{[0-9a-zA-Z_.]+}}, align 16

void __attribute__((noinline))
test_cast() {
  res = _mm_castpd_ps(md1);
  resi = _mm_castpd_si128(md1);
  resd = _mm_castps_pd(m1);
  resi = _mm_castps_si128(m1);
  resd = _mm_castsi128_pd(mi1);
  res = _mm_castsi128_ps(mi1);
}

// CHECK-LABEL: @test_cast

// CHECK-LABEL: define available_externally <4 x float> @_mm_castpd_ps
// CHECK: bitcast <2 x double> %{{[0-9a-zA-Z_.]+}} to <4 x float>

// CHECK-LABEL: define available_externally <2 x i64> @_mm_castpd_si128
// CHECK: bitcast <2 x double> %{{[0-9a-zA-Z_.]+}} to <2 x i64>

// CHECK-LABEL: define available_externally <2 x double> @_mm_castps_pd
// CHECK: bitcast <4 x float> %{{[0-9a-zA-Z_.]+}} to <2 x double>

// CHECK-LABEL: define available_externally <2 x i64> @_mm_castps_si128
// CHECK: bitcast <4 x float> %{{[0-9a-zA-Z_.]+}} to <2 x i64>

// CHECK-LABEL: define available_externally <2 x double> @_mm_castsi128_pd
// CHECK: bitcast <2 x i64> %{{[0-9a-zA-Z_.]+}} to <2 x double>

// CHECK-LABEL: define available_externally <4 x float> @_mm_castsi128_ps
// CHECK: bitcast <2 x i64> %{{[0-9a-zA-Z_.]+}} to <4 x float>

void __attribute__((noinline))
test_cmp() {
  resi = _mm_cmpeq_epi32(mi1, mi2);
  resi = _mm_cmpeq_epi16(mi1, mi2);
  resi = _mm_cmpeq_epi8(mi1, mi2);
  resi = _mm_cmpgt_epi32(mi1, mi2);
  resi = _mm_cmpgt_epi16(mi1, mi2);
  resi = _mm_cmpgt_epi8(mi1, mi2);
  resi = _mm_cmplt_epi32(mi1, mi2);
  resi = _mm_cmplt_epi16(mi1, mi2);
  resi = _mm_cmplt_epi8(mi1, mi2);
  resd = _mm_cmpeq_pd(md1, md2);
  resd = _mm_cmpeq_sd(md1, md2);
  resd = _mm_cmpge_pd(md1, md2);
  resd = _mm_cmpge_sd(md1, md2);
  resd = _mm_cmpgt_pd(md1, md2);
  resd = _mm_cmpgt_sd(md1, md2);
  resd = _mm_cmple_pd(md1, md2);
  resd = _mm_cmple_sd(md1, md2);
  resd = _mm_cmplt_pd(md1, md2);
  resd = _mm_cmplt_sd(md1, md2);
  resd = _mm_cmpneq_pd(md1, md2);
  resd = _mm_cmpneq_sd(md1, md2);
  resd = _mm_cmpnge_pd(md1, md2);
  resd = _mm_cmpnge_sd(md1, md2);
  resd = _mm_cmpngt_pd(md1, md2);
  resd = _mm_cmpngt_sd(md1, md2);
  resd = _mm_cmpnle_pd(md1, md2);
  resd = _mm_cmpnle_sd(md1, md2);
  resd = _mm_cmpnlt_pd(md1, md2);
  resd = _mm_cmpnlt_sd(md1, md2);
  resd = _mm_cmpord_pd(md1, md2);
  resd = _mm_cmpord_sd(md1, md2);
  resd = _mm_cmpunord_pd(md1, md2);
  resd = _mm_cmpunord_sd(md1, md2);
}

// CHECK-LABEL: @test_cmp

// CHECK-LABEL: define available_externally <2 x i64> @_mm_cmpeq_epi32
// CHECK: call <4 x i32> @vec_cmpeq(int vector[4], int vector[4])

// CHECK-LABEL: define available_externally <2 x i64> @_mm_cmpeq_epi16
// CHECK: call <8 x i16> @vec_cmpeq(short vector[8], short vector[8])

// CHECK-LABEL: define available_externally <2 x i64> @_mm_cmpeq_epi8
// CHECK: call <16 x i8> @vec_cmpeq(signed char vector[16], signed char vector[16])

// CHECK-LABEL: define available_externally <2 x i64> @_mm_cmpgt_epi32
// CHECK: call <4 x i32> @vec_cmpgt(int vector[4], int vector[4])

// CHECK-LABEL: define available_externally <2 x i64> @_mm_cmpgt_epi16
// CHECK: call <8 x i16> @vec_cmpgt(short vector[8], short vector[8])

// CHECK-LABEL: define available_externally <2 x i64> @_mm_cmpgt_epi8
// CHECK: call <16 x i8> @vec_cmpgt(signed char vector[16], signed char vector[16])

// CHECK-LABEL: define available_externally <2 x i64> @_mm_cmplt_epi32
// CHECK: call <4 x i32> @vec_cmplt(int vector[4], int vector[4])

// CHECK-LABEL: define available_externally <2 x i64> @_mm_cmplt_epi16
// CHECK: call <8 x i16> @vec_cmplt(short vector[8], short vector[8])

// CHECK-LABEL: define available_externally <2 x i64> @_mm_cmplt_epi8
// CHECK: call <16 x i8> @vec_cmplt(signed char vector[16], signed char vector[16])

// CHECK-LABEL: define available_externally <2 x double> @_mm_cmpeq_pd
// CHECK: call <2 x i64> @vec_cmpeq(double vector[2], double vector[2])

// CHECK-LABEL: define available_externally <2 x double> @_mm_cmpeq_sd
// CHECK: call <2 x double> @vec_splats(double)
// CHECK: call <2 x double> @vec_splats(double)
// CHECK: call <2 x i64> @vec_cmpeq(double vector[2], double vector[2])
// CHECK: call <2 x double> @_mm_setr_pd(double noundef %{{[0-9a-zA-Z_.]+}}, double noundef %{{[0-9a-zA-Z_.]+}})

// CHECK-LABEL: define available_externally <2 x double> @_mm_cmpge_pd
// CHECK: call <2 x i64> @vec_cmpge(double vector[2], double vector[2])

// CHECK-LABEL: define available_externally <2 x double> @_mm_cmpge_sd
// CHECK: call <2 x double> @vec_splats(double)
// CHECK: call <2 x double> @vec_splats(double)
// CHECK: call <2 x i64> @vec_cmpge(double vector[2], double vector[2])
// CHECK: call <2 x double> @_mm_setr_pd(double noundef %{{[0-9a-zA-Z_.]+}}, double noundef %{{[0-9a-zA-Z_.]+}})

// CHECK-LABEL: define available_externally <2 x double> @_mm_cmpgt_pd
// CHECK: call <2 x i64> @vec_cmpgt(double vector[2], double vector[2])

// CHECK-LABEL: define available_externally <2 x double> @_mm_cmpgt_sd
// CHECK: call <2 x double> @vec_splats(double)
// CHECK: call <2 x double> @vec_splats(double)
// CHECK: call <2 x i64> @vec_cmpgt(double vector[2], double vector[2])
// CHECK: call <2 x double> @_mm_setr_pd(double noundef %{{[0-9a-zA-Z_.]+}}, double noundef %{{[0-9a-zA-Z_.]+}})

// CHECK-LABEL: define available_externally <2 x double> @_mm_cmple_pd
// CHECK: call <2 x i64> @vec_cmple(double vector[2], double vector[2])

// CHECK-LABEL: define available_externally <2 x double> @_mm_cmple_sd
// CHECK: call <2 x double> @vec_splats(double)
// CHECK: call <2 x double> @vec_splats(double)
// CHECK: call <2 x i64> @vec_cmple(double vector[2], double vector[2])
// CHECK: call <2 x double> @_mm_setr_pd(double noundef %{{[0-9a-zA-Z_.]+}}, double noundef %{{[0-9a-zA-Z_.]+}})

// CHECK-LABEL: define available_externally <2 x double> @_mm_cmplt_pd
// CHECK: call <2 x i64> @vec_cmplt(double vector[2], double vector[2])

// CHECK-LABEL: define available_externally <2 x double> @_mm_cmplt_sd
// CHECK: call <2 x double> @vec_splats(double)
// CHECK: call <2 x double> @vec_splats(double)
// CHECK: call <2 x i64> @vec_cmplt(double vector[2], double vector[2])
// CHECK: call <2 x double> @_mm_setr_pd(double noundef %{{[0-9a-zA-Z_.]+}}, double noundef %{{[0-9a-zA-Z_.]+}})

// CHECK-LABEL: define available_externally <2 x double> @_mm_cmpneq_pd
// CHECK: call <2 x i64> @vec_cmpeq(double vector[2], double vector[2])
// CHECK: call <2 x double> @vec_nor(double vector[2], double vector[2])

// CHECK-LABEL: define available_externally <2 x double> @_mm_cmpneq_sd
// CHECK: call <2 x double> @vec_splats(double)
// CHECK: call <2 x double> @vec_splats(double)
// CHECK: call <2 x i64> @vec_cmpeq(double vector[2], double vector[2])
// CHECK: call <2 x double> @vec_nor(double vector[2], double vector[2])
// CHECK: call <2 x double> @_mm_setr_pd(double noundef %{{[0-9a-zA-Z_.]+}}, double noundef %{{[0-9a-zA-Z_.]+}})

// CHECK-LABEL: define available_externally <2 x double> @_mm_cmpnge_pd
// CHECK: call <2 x i64> @vec_cmplt(double vector[2], double vector[2])

// CHECK-LABEL: define available_externally <2 x double> @_mm_cmpnge_sd
// CHECK: call <2 x double> @vec_splats(double)
// CHECK: call <2 x double> @vec_splats(double)
// CHECK: call <2 x i64> @vec_cmplt(double vector[2], double vector[2])
// CHECK: call <2 x double> @_mm_setr_pd(double noundef %{{[0-9a-zA-Z_.]+}}, double noundef %{{[0-9a-zA-Z_.]+}})

// CHECK-LABEL: define available_externally <2 x double> @_mm_cmpngt_pd
// CHECK: call <2 x i64> @vec_cmple(double vector[2], double vector[2])

// CHECK-LABEL: define available_externally <2 x double> @_mm_cmpngt_sd
// CHECK: call <2 x double> @vec_splats(double)
// CHECK: call <2 x double> @vec_splats(double)
// CHECK: call <2 x i64> @vec_cmple(double vector[2], double vector[2])
// CHECK: call <2 x double> @_mm_setr_pd(double noundef %{{[0-9a-zA-Z_.]+}}, double noundef %{{[0-9a-zA-Z_.]+}})

// CHECK-LABEL: define available_externally <2 x double> @_mm_cmpnle_pd
// CHECK: call <2 x i64> @vec_cmpgt(double vector[2], double vector[2])

// CHECK-LABEL: define available_externally <2 x double> @_mm_cmpnle_sd
// CHECK: call <2 x double> @vec_splats(double)
// CHECK: call <2 x double> @vec_splats(double)
// CHECK: call <2 x i64> @vec_cmpge(double vector[2], double vector[2])
// CHECK: call <2 x double> @_mm_setr_pd(double noundef %{{[0-9a-zA-Z_.]+}}, double noundef %{{[0-9a-zA-Z_.]+}})

// CHECK-LABEL: define available_externally <2 x double> @_mm_cmpnlt_pd
// CHECK: call <2 x i64> @vec_cmpge(double vector[2], double vector[2])

// CHECK-LABEL: define available_externally <2 x double> @_mm_cmpnlt_sd
// CHECK: call <2 x double> @vec_splats(double)
// CHECK: call <2 x double> @vec_splats(double)
// CHECK: call <2 x i64> @vec_cmpge(double vector[2], double vector[2])
// CHECK: call <2 x double> @_mm_setr_pd(double noundef %{{[0-9a-zA-Z_.]+}}, double noundef %{{[0-9a-zA-Z_.]+}})

// CHECK-LABEL: define available_externally <2 x double> @_mm_cmpord_pd
// CHECK: call <2 x i64> @vec_cmpeq(double vector[2], double vector[2])
// CHECK: call <2 x i64> @vec_cmpeq(double vector[2], double vector[2])
// CHECK: call <2 x i64> @vec_and(unsigned long long vector[2], unsigned long long vector[2])

// CHECK-LABEL: define available_externally <2 x double> @_mm_cmpord_sd
// CHECK: call <2 x double> @vec_splats(double)
// CHECK: call <2 x double> @vec_splats(double)
// CHECK: call <2 x double> @_mm_cmpord_pd(<2 x double> noundef %{{[0-9a-zA-Z_.]+}}, <2 x double> noundef %{{[0-9a-zA-Z_.]+}})
// CHECK: call <2 x double> @_mm_setr_pd(double noundef %{{[0-9a-zA-Z_.]+}}, double noundef %{{[0-9a-zA-Z_.]+}})

// CHECK-LABEL: define available_externally <2 x double> @_mm_cmpunord_pd
// CHECK: call <2 x i64> @vec_cmpeq(double vector[2], double vector[2])
// CHECK: call <2 x i64> @vec_cmpeq(double vector[2], double vector[2])
// CHECK: call <2 x i64> @vec_nor(unsigned long long vector[2], unsigned long long vector[2])
// CHECK: call <2 x i64> @vec_orc(unsigned long long vector[2], unsigned long long vector[2])

// CHECK-LABEL: define available_externally <2 x double> @_mm_cmpunord_sd
// CHECK: call <2 x double> @vec_splats(double)
// CHECK: call <2 x double> @vec_splats(double)
// CHECK: call <2 x double> @_mm_cmpunord_pd(<2 x double> noundef %{{[0-9a-zA-Z_.]+}}, <2 x double> noundef %{{[0-9a-zA-Z_.]+}})
// CHECK: call <2 x double> @_mm_setr_pd(double noundef %{{[0-9a-zA-Z_.]+}}, double noundef %{{[0-9a-zA-Z_.]+}})

void __attribute__((noinline))
test_comi() {
  i = _mm_comieq_sd(md1, md2);
  i = _mm_comige_sd(md1, md2);
  i = _mm_comigt_sd(md1, md2);
  i = _mm_comile_sd(md1, md2);
  i = _mm_comilt_sd(md1, md2);
  i = _mm_comineq_sd(md1, md2);
}

// CHECK-LABEL: @test_comi

// CHECK-LABEL: define available_externally signext i32 @_mm_comieq_sd
// CHECK: %[[CMP:[0-9a-zA-Z_.]+]] = fcmp oeq double
// CHECK: zext i1 %[[CMP]] to i32

// CHECK-LABEL: define available_externally signext i32 @_mm_comige_sd
// CHECK: %[[CMP:[0-9a-zA-Z_.]+]] = fcmp oge double
// CHECK: zext i1 %[[CMP]] to i32

// CHECK-LABEL: define available_externally signext i32 @_mm_comigt_sd
// CHECK: %[[CMP:[0-9a-zA-Z_.]+]] = fcmp ogt double
// CHECK: zext i1 %[[CMP]] to i32

// CHECK-LABEL: define available_externally signext i32 @_mm_comile_sd
// CHECK: %[[CMP:[0-9a-zA-Z_.]+]] = fcmp ole double
// CHECK: zext i1 %[[CMP]] to i32

// CHECK-LABEL: define available_externally signext i32 @_mm_comilt_sd
// CHECK: %[[CMP:[0-9a-zA-Z_.]+]] = fcmp olt double
// CHECK: zext i1 %[[CMP]] to i32

// CHECK-LABEL: define available_externally signext i32 @_mm_comineq_sd
// CHECK: %[[CMP:[0-9a-zA-Z_.]+]] = fcmp une double
// CHECK: zext i1 %[[CMP]] to i32

void __attribute__((noinline))
test_control() {
  _mm_clflush(dp);
  _mm_lfence();
  _mm_mfence();
  _mm_pause();
}

// CHECK-LABEL: @test_control

// CHECK-LABEL: define available_externally void @_mm_clflush
// CHECK: call void asm sideeffect "dcbf 0,$0", "b,~{memory}"(i8* %{{[0-9a-zA-Z_.]+}})

// CHECK-LABEL: define available_externally void @_mm_lfence()
// CHECK: fence release

// CHECK-LABEL: define available_externally void @_mm_mfence()
// CHECK: fence seq_cst

// CHECK-LABEL: define available_externally void @_mm_pause()
// CHECK: call i64 asm sideeffect "\09mfppr\09$0;   or 31,31,31;   isync;   lwsync;   isync;   mtppr\09$0;", "=r,~{memory}"()

void __attribute__((noinline))
test_converts() {
  resd = _mm_cvtepi32_pd(mi1);
  res = _mm_cvtepi32_ps(mi1);
  resi = _mm_cvtpd_epi32(md1);
  res64 = _mm_cvtpd_pi32(md1);
  res = _mm_cvtpd_ps(md1);
  resd = _mm_cvtpi32_pd(res64);
  resi = _mm_cvtps_epi32(m1);
  resd = _mm_cvtps_pd(m1);
  *dp = _mm_cvtsd_f64(md1);
  i = _mm_cvtsd_si32(md1);
  i64s[0] = _mm_cvtsd_si64(md1);
  i64s[0] = _mm_cvtsd_si64x(md1);
  res = _mm_cvtsd_ss(m1, md2);
  i = _mm_cvtsi128_si32(mi1);
  i64s[0] = _mm_cvtsi128_si64(mi1);
  i64s[0] = _mm_cvtsi128_si64x(mi1);
  resd = _mm_cvtsi32_sd(md1, i);
  resi = _mm_cvtsi32_si128(i);
  resd = _mm_cvtsi64_sd(md1, i64s[1]);
  resi = _mm_cvtsi64_si128(i64s[1]);
  resd = _mm_cvtsi64x_sd(md1, i64s[1]);
  resi = _mm_cvtsi64x_si128(i64s[1]);
  resd = _mm_cvtss_sd(md1, m1);
  resi = _mm_cvttpd_epi32(md1);
  res64 = _mm_cvttpd_pi32(md1);
  resi = _mm_cvttps_epi32(m1);
  i = _mm_cvttsd_si32(md1);
  i64s[0] = _mm_cvttsd_si64(md1);
  i64s[0] = _mm_cvttsd_si64x(md1);
}

// CHECK-LABEL: @test_converts

// CHECK-LABEL: define available_externally <2 x double> @_mm_cvtepi32_pd
// CHECK: call <2 x i64> @vec_unpackh(int vector[4])
// CHECK: %[[CONV:[0-9a-zA-Z_.]+]] = sitofp <2 x i64> %{{[0-9a-zA-Z_.]+}} to <2 x double>
// CHECK: fmul <2 x double> %[[CONV]], <double 1.000000e+00, double 1.000000e+00>

// CHECK-LABEL: define available_externally <4 x float> @_mm_cvtepi32_ps
// CHECK: call <4 x float> @llvm.ppc.altivec.vcfsx(<4 x i32> %{{[0-9a-zA-Z_.]+}}, i32 0)

// CHECK-LABEL: define available_externally <2 x i64> @_mm_cvtpd_epi32
// CHECK: call <2 x double> @vec_rint(double vector[2])
// CHECK: store <4 x i32> zeroinitializer, <4 x i32>* %{{[0-9a-zA-Z_.]+}}, align 16
// CHECK: call <4 x i32> asm "xvcvdpsxws ${0:x},${1:x}", "=^wa,^wa"(<2 x double> %{{[0-9a-zA-Z_.]+}})
// CHECK-LE: call <4 x i32> @vec_mergeo(int vector[4], int vector[4])
// CHECK-BE: call <4 x i32> @vec_mergee(int vector[4], int vector[4])
// CHECK: call <4 x i32> @vec_vpkudum(long long vector[2], long long vector[2])(<2 x i64> noundef %{{[0-9a-zA-Z_.]+}}, <2 x i64> noundef zeroinitializer)

// CHECK-LABEL: define available_externally i64 @_mm_cvtpd_pi32
// CHECK: call <2 x i64> @_mm_cvtpd_epi32(<2 x double> noundef %{{[0-9a-zA-Z_.]+}})
// CHECK: extractelement <2 x i64> %{{[0-9a-zA-Z_.]+}}, i32 0

// CHECK-LABEL: define available_externally <4 x float> @_mm_cvtpd_ps
// CHECK: store <4 x i32> zeroinitializer, <4 x i32>* %{{[0-9a-zA-Z_.]+}}, align 16
// CHECK: call <4 x i32> asm "xvcvdpsp ${0:x},${1:x}", "=^wa,^wa"(<2 x double> %{{[0-9a-zA-Z_.]+}})
// CHECK-LE: call <4 x i32> @vec_mergeo(int vector[4], int vector[4])
// CHECK-BE: call <4 x i32> @vec_mergee(int vector[4], int vector[4])
// CHECK: call <4 x i32> @vec_vpkudum(long long vector[2], long long vector[2])(<2 x i64> noundef %{{[0-9a-zA-Z_.]+}}, <2 x i64> noundef zeroinitializer)

// CHECK-LABEL: define available_externally <2 x double> @_mm_cvtpi32_pd
// CHECK: call <2 x i64> @vec_splats(unsigned long long)
// CHECK: call <2 x i64> @vec_unpackl(int vector[4])
// CHECK: %[[CONV:[0-9a-zA-Z_.]+]] = sitofp <2 x i64> %{{[0-9a-zA-Z._]+}} to <2 x double>
// CHECK: fmul <2 x double> %[[CONV]], <double 1.000000e+00, double 1.000000e+00>

// CHECK-LABEL: define available_externally <2 x i64> @_mm_cvtps_epi32
// CHECK: call <4 x float> @vec_rint(float vector[4])
// CHECK: call <4 x i32> @llvm.ppc.altivec.vctsxs(<4 x float> %{{[0-9a-zA-Z_.]+}}, i32 0)

// CHECK-LABEL: define available_externally <2 x double> @_mm_cvtps_pd
// CHECK-BE: call <4 x float> @vec_vmrghw(float vector[4], float vector[4])
// CHECK-BE: call <2 x double> asm " xvcvspdp ${0:x},${1:x}", "=^wa,^wa"(<4 x float> %{{[0-9a-zA-Z_.]+}})
// CHECK-LE: shufflevector <4 x i32> %{{[0-9a-zA-Z_.]+}}, <4 x i32> %{{[0-9a-zA-Z_.]+}}, <4 x i32> <i32 5, i32 6, i32 7, i32 0>
// CHECK-LE: shufflevector <4 x i32> %{{[0-9a-zA-Z_.]+}}, <4 x i32> %{{[0-9a-zA-Z_.]+}}, <4 x i32> <i32 6, i32 7, i32 0, i32 1>
// CHECK-LE: call <2 x double> asm " xvcvspdp ${0:x},${1:x}", "=^wa,^wa"(<4 x float> %{{[0-9a-zA-Z_.]+}})

// CHECK-LABEL: define available_externally double @_mm_cvtsd_f64
// CHECK: extractelement <2 x double> %{{[0-9a-zA-Z_.]+}}, i32 0

// CHECK-LABEL: define available_externally signext i32 @_mm_cvtsd_si32
// CHECK: call <2 x double> @vec_rint(double vector[2])
// CHECK: fptosi double %{{[0-9a-zA-Z_.]+}} to i32

// CHECK-LABEL: define available_externally i64 @_mm_cvtsd_si64
// CHECK: call <2 x double> @vec_rint(double vector[2])
// CHECK: fptosi double %{{[0-9a-zA-Z_.]+}} to i64

// CHECK-LABEL: define available_externally i64 @_mm_cvtsd_si64x
// CHECK: call i64 @_mm_cvtsd_si64(<2 x double> noundef %{{[0-9a-zA-Z_.]+}})

// CHECK-LABEL: define available_externally <4 x float> @_mm_cvtsd_ss
// CHECK-BE: %[[EXT:[0-9a-zA-Z_.]+]] = extractelement <2 x double> %{{[0-9a-zA-Z_.]+}}, i32 0
// CHECK-BE: %[[TRUNC:[0-9a-zA-Z_.]+]] = fptrunc double %[[EXT]] to float
// CHECK-BE: insertelement <4 x float> %{{[0-9a-zA-Z_.]+}}, float %[[TRUNC]], i32 0
// CHECK-LE: call <2 x double> @vec_splat(double vector[2], unsigned int)(<2 x double> noundef %{{[0-9a-zA-Z_.]+}}, i32 noundef zeroext 0)
// CHECK-LE: shufflevector <4 x i32> %{{[0-9a-zA-Z_.]+}}, <4 x i32> %{{[0-9a-zA-Z_.]+}}, <4 x i32> <i32 5, i32 6, i32 7, i32 0>
// CHECK-LE: call <4 x float> asm "xscvdpsp ${0:x},${1:x}", "=^wa,^wa"(<2 x double> %{{[0-9a-zA-Z_.]+}})
// CHECK-LE: shufflevector <4 x i32> %{{[0-9a-zA-Z_.]+}}, <4 x i32> %{{[0-9a-zA-Z_.]+}}, <4 x i32> <i32 7, i32 0, i32 1, i32 2>

// CHECK-LABEL: define available_externally signext i32 @_mm_cvtsi128_si32
// CHECK: extractelement <4 x i32> %{{[0-9a-zA-Z_.]+}}, i32 0

// CHECK-LABEL: define available_externally i64 @_mm_cvtsi128_si64
// CHECK: extractelement <2 x i64> %{{[0-9a-zA-Z_.]+}}, i32 0

// CHECK-LABEL: define available_externally i64 @_mm_cvtsi128_si64x
// CHECK: extractelement <2 x i64> %{{[0-9a-zA-Z_.]+}}, i32 0

// CHECK-LABEL: define available_externally <2 x double> @_mm_cvtsi32_sd
// CHECK: sitofp i32 %{{[0-9a-zA-Z_.]+}} to double

// CHECK-LABEL: define available_externally <2 x i64> @_mm_cvtsi32_si128
// CHECK: call <2 x i64> @_mm_set_epi32(i32 noundef signext 0, i32 noundef signext 0, i32 noundef signext 0, i32 noundef signext %{{[0-9a-zA-Z_.]+}})

// CHECK-LABEL: define available_externally <2 x double> @_mm_cvtsi64_sd
// CHECK: sitofp i64 %{{[0-9a-zA-Z_.]+}} to double

// CHECK-LABEL: define available_externally <2 x i64> @_mm_cvtsi64_si128
// CHECK: %[[INS:[0-9a-zA-Z_.]+]] = insertelement <2 x i64> undef, i64 %{{[0-9a-zA-Z_.]+}}, i32 0
// CHECK: insertelement <2 x i64> %[[INS]], i64 0, i32 1

// CHECK-LABEL: define available_externally <2 x double> @_mm_cvtsi64x_sd
// CHECK: call <2 x double> @_mm_cvtsi64_sd(<2 x double> noundef %{{[0-9a-zA-Z_.]+}}, i64 noundef %{{[0-9a-zA-Z_.]+}})

// CHECK-LABEL: define available_externally <2 x i64> @_mm_cvtsi64x_si128
// CHECK: %[[INS:[0-9a-zA-Z_.]+]] = insertelement <2 x i64> undef, i64 %{{[0-9a-zA-Z_.]+}}, i32 0
// CHECK: insertelement <2 x i64> %[[INS]], i64 0, i32 1

// CHECK-LABEL: define available_externally <2 x double> @_mm_cvtss_sd
// CHECK-BE: fpext float %{{[0-9a-zA-Z_.]+}} to double
// CHECK-LE: call <4 x float> @vec_splat(float vector[4], unsigned int)(<4 x float> noundef %{{[0-9a-zA-Z_.]+}}, i32 noundef zeroext 0)
// CHECK-LE: call <2 x double> asm "xscvspdp ${0:x},${1:x}", "=^wa,^wa"(<4 x float> %{{[0-9a-zA-Z_.]+}})
// CHECK-LE: call <2 x double> @vec_mergel(double vector[2], double vector[2])

// CHECK-LABEL: define available_externally <2 x i64> @_mm_cvttpd_epi32
// CHECK: call <4 x i32> asm "xvcvdpsxws ${0:x},${1:x}", "=^wa,^wa"
// CHECK-LE: call <4 x i32> @vec_mergeo(int vector[4], int vector[4])
// CHECK-BE: call <4 x i32> @vec_mergee(int vector[4], int vector[4])
// CHECK: call <4 x i32> @vec_vpkudum(long long vector[2], long long vector[2])(<2 x i64> noundef %{{[0-9a-zA-Z_.]+}}, <2 x i64> noundef zeroinitializer)

// CHECK-LABEL: define available_externally i64 @_mm_cvttpd_pi32
// CHECK: call <2 x i64> @_mm_cvttpd_epi32(<2 x double> noundef %{{[0-9a-zA-Z_.]+}})

// CHECK-LABEL: define available_externally <2 x i64> @_mm_cvttps_epi32
// CHECK: call <4 x i32> @llvm.ppc.altivec.vctsxs(<4 x float> %{{[0-9a-zA-Z_.]+}}, i32 0)

// CHECK-LABEL: define available_externally signext i32 @_mm_cvttsd_si32
// CHECK: fptosi double %{{[0-9a-zA-Z_.]+}} to i32

// CHECK-LABEL: define available_externally i64 @_mm_cvttsd_si64
// CHECK: fptosi double %{{[0-9a-zA-Z_.]+}} to i64

// CHECK-LABEL: define available_externally i64 @_mm_cvttsd_si64x
// CHECK: call i64 @_mm_cvttsd_si64(<2 x double> noundef %{{[0-9a-zA-Z_.]+}})

void __attribute__((noinline))
test_div() {
  resd = _mm_div_pd(md1, md2);
  resd = _mm_div_sd(md1, md2);
}

// CHECK-LABEL: @test_div

// CHECK-LABEL: define available_externally <2 x double> @_mm_div_pd
// CHECK: fdiv <2 x double>

// CHECK-LABEL: define available_externally <2 x double> @_mm_div_sd
// CHECK: fdiv double

void __attribute__((noinline))
test_extract() {
  i = _mm_extract_epi16(mi1, i);
}

// CHECK-LABEL: @test_extract

// CHECK-LABEL: define available_externally signext i32 @_mm_extract_epi16
// CHECK: %[[AND:[0-9a-zA-Z_.]+]] = and i32 %{{[0-9a-zA-Z_.]+}}, 7
// CHECK: %[[EXT:[0-9a-zA-Z_.]+]] = extractelement <8 x i16> %{{[0-9a-zA-Z_.]+}}, i32 %[[AND]]
// CHECK: zext i16 %[[EXT]] to i32

void __attribute__((noinline))
test_insert() {
  resi = _mm_insert_epi16 (mi1, i, is[0]);
}

// CHECK-LABEL: @test_insert

// CHECK-LABEL: define available_externally <2 x i64> @_mm_insert_epi16
// CHECK: trunc i32 %{{[0-9a-zA-Z_.]+}} to i16
// CHECK: and i32 %{{[0-9a-zA-Z_.]+}}, 7

void __attribute__((noinline))
test_load() {
  resd = _mm_load_pd(dp);
  resd = _mm_load_pd1(dp);
  resd = _mm_load_sd(dp);
  resi = _mm_load_si128(mip);
  resd = _mm_load1_pd(dp);
  resd = _mm_loadh_pd(md1, dp);
  resi = _mm_loadl_epi64(mip);
  resd = _mm_loadl_pd(md1, dp);
  resd = _mm_loadr_pd(dp);
  resd = _mm_loadu_pd(dp);
  resi = _mm_loadu_si128(mip);
}

// CHECK-LABEL: @test_load

// CHECK-LABEL: define available_externally <2 x double> @_mm_load_pd
// CHECK: call <16 x i8> @vec_ld(long, unsigned char vector[16] const*)(i64 noundef 0, <16 x i8>* noundef %{{[0-9a-zA-Z_.]+}})

// CHECK-LABEL: define available_externally <2 x double> @_mm_load_pd1
// CHECK: call <2 x double> @_mm_load1_pd(double* noundef %{{[0-9a-zA-Z_.]+}})

// CHECK-LABEL: define available_externally <2 x double> @_mm_load_sd
// CHECK: call <2 x double> @_mm_set_sd(double noundef %{{[0-9a-zA-Z_.]+}})

// CHECK-LABEL: define available_externally <2 x i64> @_mm_load_si128
// CHECK: %[[ADDR:[0-9a-zA-Z_.]+]] = load <2 x i64>*, <2 x i64>** %{{[0-9a-zA-Z_.]+}}, align 8
// CHECK: load <2 x i64>, <2 x i64>* %[[ADDR]], align 16

// CHECK-LABEL: define available_externally <2 x double> @_mm_load1_pd
// CHECK: %[[ADDR:[0-9a-zA-Z_.]+]] = load double*, double** %{{[0-9a-zA-Z_.]+}}, align 8
// CHECK: %[[VAL:[0-9a-zA-Z_.]+]] = load double, double* %[[ADDR]], align 8
// CHECK: call <2 x double> @vec_splats(double)(double noundef %[[VAL]])

// CHECK-LABEL: define available_externally <2 x double> @_mm_loadh_pd
// CHECK: %[[ADDR:[0-9a-zA-Z_.]+]] = load double*, double** %{{[0-9a-zA-Z_.]+}}, align 8
// CHECK: %[[VAL:[0-9a-zA-Z_.]+]] = load double, double* %{{[0-9a-zA-Z_.]+}}, align 8
// CHECK: %[[VEC:[0-9a-zA-Z_.]+]] = load <2 x double>, <2 x double>* %{{[0-9a-zA-Z_.]+}}, align 16
// CHECK: insertelement <2 x double> %[[VEC]], double %[[VAL]], i32 1

// CHECK-LABEL: define available_externally <2 x i64> @_mm_loadl_epi64
// CHECK: call <2 x i64> @_mm_set_epi64(i64 noundef 0, i64 noundef %{{[0-9a-zA-Z_.]+}})

// CHECK-LABEL: define available_externally <2 x double> @_mm_loadl_pd
// CHECK: %[[ADDR:[0-9a-zA-Z_.]+]] = load double*, double** %{{[0-9a-zA-Z_.]+}}, align 8
// CHECK: %[[ADDR2:[0-9a-zA-Z_.]+]] = load double, double* %[[ADDR]], align 8
// CHECK: %[[VEC:[0-9a-zA-Z_.]+]] = load <2 x double>, <2 x double>* %{{[0-9a-zA-Z_.]+}}, align 16
// CHECK: insertelement <2 x double> %[[VEC]], double %[[ADDR2]], i32 0

// CHECK-LABEL: define available_externally <2 x double> @_mm_loadr_pd
// CHECK: %[[ADDR:[0-9a-zA-Z_.]+]] = load double*, double** %{{[0-9a-zA-Z_.]+}}, align 8
// CHECK: call <2 x double> @_mm_load_pd(double* noundef %[[ADDR]])
// CHECK: shufflevector <2 x i64> %{{[0-9a-zA-Z_.]+}}, <2 x i64> %{{[0-9a-zA-Z_.]+}}, <2 x i32> <i32 1, i32 2>

// CHECK-LABEL: define available_externally <2 x double> @_mm_loadu_pd
// CHECK: %[[ADDR:[0-9a-zA-Z_.]+]] = load double*, double** %{{[0-9a-zA-Z_.]+}}, align 8
// CHECK: call <2 x double> @vec_vsx_ld(int, double const*)(i32 noundef signext 0, double* noundef %[[ADDR]])

// CHECK-LABEL: define available_externally <2 x i64> @_mm_loadu_si128
// CHECK: load <2 x i64>*, <2 x i64>** %{{[0-9a-zA-Z_.]+}}, align 8
// CHECK: call <4 x i32> @vec_vsx_ld(int, int const*)(i32 noundef signext 0, i32* noundef %{{[0-9a-zA-Z_.]+}})

void __attribute__((noinline))
test_logical() {
  resd = _mm_and_pd(md1, md2);
  resi = _mm_and_si128(mi1, mi2);
  resd = _mm_andnot_pd(md1, md2);
  resi = _mm_andnot_si128(mi1, mi2);
  resd = _mm_xor_pd(md1, md2);
  resi = _mm_xor_si128(mi1, mi2);
  resd = _mm_or_pd(md1, md2);
  resi = _mm_or_si128(mi1, mi2);
}

// CHECK-LABEL: @test_logical

// CHECK-LABEL: define available_externally <2 x double> @_mm_and_pd
// CHECK: call <2 x double> @vec_and(double vector[2], double vector[2])

// CHECK-LABEL: define available_externally <2 x i64> @_mm_and_si128
// CHECK: call <2 x i64> @vec_and(long long vector[2], long long vector[2])

// CHECK-LABEL: define available_externally <2 x double> @_mm_andnot_pd
// CHECK: call <2 x double> @vec_andc(double vector[2], double vector[2])

// CHECK-LABEL: define available_externally <2 x i64> @_mm_andnot_si128
// CHECK: call <2 x i64> @vec_andc(long long vector[2], long long vector[2])

// CHECK-LABEL: define available_externally <2 x double> @_mm_xor_pd
// CHECK: call <2 x double> @vec_xor(double vector[2], double vector[2])

// CHECK-LABEL: define available_externally <2 x i64> @_mm_xor_si128
// CHECK: call <2 x i64> @vec_xor(long long vector[2], long long vector[2])

// CHECK-LABEL: define available_externally <2 x double> @_mm_or_pd
// CHECK: call <2 x double> @vec_or(double vector[2], double vector[2])

// CHECK-LABEL: define available_externally <2 x i64> @_mm_or_si128
// CHECK: call <2 x i64> @vec_or(long long vector[2], long long vector[2])

void __attribute__((noinline))
test_max() {
  resi = _mm_max_epi16(mi1, mi2);
  resi = _mm_max_epu8(mi1, mi2);
  resd = _mm_max_pd(md1, md2);
  resd = _mm_max_sd(md1, md2);
}

// CHECK-LABEL: @test_max

// CHECK-LABEL: define available_externally <2 x i64> @_mm_max_epi16
// CHECK: call <8 x i16> @vec_max(short vector[8], short vector[8])

// CHECK-LABEL: define available_externally <2 x i64> @_mm_max_epu8
// CHECK: call <16 x i8> @vec_max(unsigned char vector[16], unsigned char vector[16])

// CHECK-LABEL: define available_externally <2 x double> @_mm_max_pd
// CHECK: call <2 x double> @vec_max(double vector[2], double vector[2])

// CHECK-LABEL: define available_externally <2 x double> @_mm_max_sd
// CHECK: call <2 x double> @vec_splats(double)
// CHECK: call <2 x double> @vec_splats(double)
// CHECK: call <2 x double> @vec_max(double vector[2], double vector[2])
// CHECK: call <2 x double> @_mm_setr_pd(double noundef %{{[0-9a-zA-Z_.]+}}, double noundef %{{[0-9a-zA-Z_.]+}})

void __attribute__((noinline))
test_min() {
  resi = _mm_min_epi16(mi1, mi2);
  resi = _mm_min_epu8(mi1, mi2);
  resd = _mm_min_pd(md1, md2);
  resd = _mm_min_sd(md1, md2);
}

// CHECK-LABEL: @test_min

// CHECK-LABEL: define available_externally <2 x i64> @_mm_min_epi16
// CHECK: call <8 x i16> @vec_min(short vector[8], short vector[8])

// CHECK-LABEL: define available_externally <2 x i64> @_mm_min_epu8
// CHECK: call <16 x i8> @vec_min(unsigned char vector[16], unsigned char vector[16])

// CHECK-LABEL: define available_externally <2 x double> @_mm_min_pd
// CHECK: call <2 x double> @vec_min(double vector[2], double vector[2])

// CHECK-LABEL: define available_externally <2 x double> @_mm_min_sd
// CHECK: call <2 x double> @vec_splats(double)
// CHECK: call <2 x double> @vec_splats(double)
// CHECK: call <2 x double> @vec_min(double vector[2], double vector[2])
// CHECK: call <2 x double> @_mm_setr_pd(double noundef %{{[0-9a-zA-Z_.]+}}, double noundef %{{[0-9a-zA-Z_.]+}})

void __attribute__((noinline))
test_move() {
  resi = _mm_move_epi64(mi1);
  resd = _mm_move_sd(md1, md2);
  i = _mm_movemask_epi8(mi1);
  i = _mm_movemask_pd(md1);
  res64 = _mm_movepi64_pi64(mi1);
  resi = _mm_movpi64_epi64(m641);
  _mm_maskmoveu_si128(mi1, mi2, chs);
}

// CHECK-LABEL: @test_move

// CHECK-LABEL: define available_externally <2 x i64> @_mm_move_epi64
// CHECK: call <2 x i64> @_mm_set_epi64(i64 noundef 0, i64 noundef %{{[0-9a-zA-Z_.]+}})

// CHECK-LABEL: define available_externally <2 x double> @_mm_move_sd
// CHECK: %[[EXT:[0-9a-zA-Z_.]+]] = extractelement <2 x double> %{{[0-9a-zA-Z_.]+}}, i32 0
// CHECK: insertelement <2 x double> %{{[0-9a-zA-Z_.]+}}, double %[[EXT]], i32 0

// CHECK-P10-LE-LABEL: define available_externally signext i32 @_mm_movemask_epi8
// CHECK-P10-LE: call zeroext i32 @vec_extractm(unsigned char vector[16])(<16 x i8> noundef %{{[0-9a-zA-Z_.]+}})

// CHECK-LABEL: define available_externally signext i32 @_mm_movemask_epi8
// CHECK: call <2 x i64> @vec_vbpermq(unsigned char vector[16], unsigned char vector[16])(<16 x i8> noundef %{{[0-9a-zA-Z_.]+}}, <16 x i8> noundef <i8 120, i8 112, i8 104, i8 96, i8 88, i8 80, i8 72, i8 64, i8 56, i8 48, i8 40, i8 32, i8 24, i8 16, i8 8, i8 0>)
// CHECK-LE: %[[VAL:[0-9a-zA-Z_.]+]] = extractelement <2 x i64> %{{[0-9a-zA-Z_.]+}}, i32 1
// CHECK-BE: %[[VAL:[0-9a-zA-Z_.]+]] = extractelement <2 x i64> %{{[0-9a-zA-Z_.]+}}, i32 0
// CHECK: trunc i64 %[[VAL]] to i32

// CHECK-P10-LE-LABEL: define available_externally signext i32 @_mm_movemask_pd
// CHECK-P10-LE: call zeroext i32 @vec_extractm(unsigned long long vector[2])(<2 x i64> noundef %{{[0-9a-zA-Z_.]+}})

// CHECK-LABEL: define available_externally signext i32 @_mm_movemask_pd
// CHECK-LE: call <2 x i64> @vec_vbpermq(unsigned char vector[16], unsigned char vector[16])(<16 x i8> noundef %{{[0-9a-zA-Z_.]+}}, <16 x i8> noundef bitcast (<4 x i32> <i32 -2139094976, i32 -2139062144, i32 -2139062144, i32 -2139062144> to <16 x i8>))
// CHECK-LE: extractelement <2 x i64> %{{[0-9a-zA-Z_.]+}}, i32 1
// CHECK-BE: call <2 x i64> @vec_vbpermq(unsigned char vector[16], unsigned char vector[16])(<16 x i8> noundef %{{[0-9a-zA-Z_.]+}}, <16 x i8> noundef bitcast (<4 x i32> <i32 -2139062144, i32 -2139062144, i32 -2139062144, i32 -2139078656> to <16 x i8>))
// CHECK-BE: extractelement <2 x i64> %{{[0-9a-zA-Z_.]+}}, i32 0

// CHECK-LABEL: define available_externally i64 @_mm_movepi64_pi64
// CHECK: extractelement <2 x i64> %{{[0-9a-zA-Z_.]+}}, i32 0

// CHECK-LABEL: define available_externally <2 x i64> @_mm_movpi64_epi64
// CHECK: call <2 x i64> @_mm_set_epi64(i64 noundef 0, i64 noundef %{{[0-9a-zA-Z_.]+}})

// CHECK-LABEL: define available_externally void @_mm_maskmoveu_si128
// CHECK: call <2 x i64> @_mm_loadu_si128(<2 x i64>* noundef %{{[0-9a-zA-Z_.]+}})
// CHECK: call <16 x i8> @vec_cmpgt(unsigned char vector[16], unsigned char vector[16])
// CHECK: call <16 x i8> @vec_sel(unsigned char vector[16], unsigned char vector[16], unsigned char vector[16])
// CHECK: call void @_mm_storeu_si128(<2 x i64>* noundef %{{[0-9a-zA-Z_.]+}}, <2 x i64> noundef %{{[0-9a-zA-Z_.]+}})

void __attribute__((noinline))
test_mul() {
  resi = _mm_mul_epu32(mi1, mi2);
  resd = _mm_mul_pd(md1, md2);
  resd = _mm_mul_sd(md1, md2);
  res64 = _mm_mul_su32(m641, m642);
  resi = _mm_mulhi_epi16(mi1, mi2);
  resi = _mm_mulhi_epu16(mi1, mi2);
  resi = _mm_mullo_epi16(mi1, mi2);
}

// CHECK-LABEL: @test_mul

// CHECK-LABEL: define available_externally <2 x i64> @_mm_mul_epu32
// CHECK-LE: call <2 x i64> asm "vmulouw $0,$1,$2", "=v,v,v"(<2 x i64> %{{[0-9a-zA-Z_.]+}}, <2 x i64> %{{[0-9a-zA-Z_.]+}})
// CHECK-BE: call <2 x i64> asm "vmuleuw $0,$1,$2", "=v,v,v"(<2 x i64> %{{[0-9a-zA-Z_.]+}}, <2 x i64> %{{[0-9a-zA-Z_.]+}})

// CHECK-LABEL: define available_externally <2 x double> @_mm_mul_pd
// CHECK: fmul <2 x double>

// CHECK-LABEL: define available_externally <2 x double> @_mm_mul_sd
// CHECK: fmul double

// CHECK-LABEL: define available_externally i64 @_mm_mul_su32
// CHECK: trunc i64 %{{[0-9a-zA-Z_.]+}} to i32
// CHECK: trunc i64 %{{[0-9a-zA-Z_.]+}} to i32
// CHECK: %[[EXT1:[0-9a-zA-Z_.]+]] = zext i32 %{{[0-9a-zA-Z_.]+}} to i64
// CHECK: %[[EXT2:[0-9a-zA-Z_.]+]] = zext i32 %{{[0-9a-zA-Z_.]+}} to i64
// CHECK: mul i64 %[[EXT1]], %[[EXT2]]

// CHECK-LABEL: define available_externally <2 x i64> @_mm_mulhi_epi16
// CHECK-LE: store <16 x i8> <i8 2, i8 3, i8 18, i8 19, i8 6, i8 7, i8 22, i8 23, i8 10, i8 11, i8 26, i8 27, i8 14, i8 15, i8 30, i8 31>, <16 x i8>* %{{[0-9a-zA-Z_.]+}}, align 16
// CHECK-BE: store <16 x i8> <i8 0, i8 1, i8 16, i8 17, i8 4, i8 5, i8 20, i8 21, i8 8, i8 9, i8 24, i8 25, i8 12, i8 13, i8 28, i8 29>, <16 x i8>* %{{[0-9a-zA-Z_.]+}}, align 16
// CHECK: call <4 x i32> @vec_vmulesh(<8 x i16> noundef %{{[0-9a-zA-Z_.]+}}, <8 x i16> noundef %{{[0-9a-zA-Z_.]+}})
// CHECK: call <4 x i32> @vec_vmulosh(<8 x i16> noundef %{{[0-9a-zA-Z_.]+}}, <8 x i16> noundef %{{[0-9a-zA-Z_.]+}})
// CHECK: call <4 x i32> @vec_perm(int vector[4], int vector[4], unsigned char vector[16])

// CHECK-LABEL: define available_externally <2 x i64> @_mm_mulhi_epu16
// CHECK-LE: store <16 x i8> <i8 2, i8 3, i8 18, i8 19, i8 6, i8 7, i8 22, i8 23, i8 10, i8 11, i8 26, i8 27, i8 14, i8 15, i8 30, i8 31>, <16 x i8>* %{{[0-9a-zA-Z_.]+}}, align 16
// CHECK-BE: store <16 x i8> <i8 0, i8 1, i8 16, i8 17, i8 4, i8 5, i8 20, i8 21, i8 8, i8 9, i8 24, i8 25, i8 12, i8 13, i8 28, i8 29>, <16 x i8>* %{{[0-9a-zA-Z_.]+}}, align 16
// CHECK: call <4 x i32> @vec_vmuleuh(<8 x i16> noundef %{{[0-9a-zA-Z_.]+}}, <8 x i16> noundef %{{[0-9a-zA-Z_.]+}})
// CHECK: call <4 x i32> @vec_vmulouh(<8 x i16> noundef %{{[0-9a-zA-Z_.]+}}, <8 x i16> noundef %{{[0-9a-zA-Z_.]+}})
// CHECK: call <4 x i32> @vec_perm(unsigned int vector[4], unsigned int vector[4], unsigned char vector[16])

// CHECK-LABEL: define available_externally <2 x i64> @_mm_mullo_epi16
// CHECK: mul <8 x i16>

void __attribute__((noinline))
test_pack() {
  resi = _mm_packs_epi16(mi1, mi2);
  resi = _mm_packs_epi32(mi1, mi2);
  resi = _mm_packus_epi16(mi1, mi2);
}

// CHECK-LABEL: @test_pack

// CHECK-LABEL: define available_externally <2 x i64> @_mm_packs_epi16
// CHECK: call <16 x i8> @vec_packs(short vector[8], short vector[8])

// CHECK-LABEL: define available_externally <2 x i64> @_mm_packs_epi32
// CHECK: call <8 x i16> @vec_packs(int vector[4], int vector[4])

// CHECK-LABEL: define available_externally <2 x i64> @_mm_packus_epi16
// CHECK: call <16 x i8> @vec_packsu(short vector[8], short vector[8])

void __attribute__((noinline))
test_sad() {
  resi = _mm_sad_epu8(mi1, mi2);
}

// CHECK-LABEL: @test_sad

// CHECK-LABEL: define available_externally <2 x i64> @_mm_sad_epu8
// CHECK: call <16 x i8> @vec_min(unsigned char vector[16], unsigned char vector[16])
// CHECK: call <16 x i8> @vec_max(unsigned char vector[16], unsigned char vector[16])
// CHECK: call <16 x i8> @vec_sub(unsigned char vector[16], unsigned char vector[16])
// CHECK: call <4 x i32> @vec_sum4s(unsigned char vector[16], unsigned int vector[4])(<16 x i8> noundef %{{[0-9a-zA-Z_.]+}}, <4 x i32> noundef zeroinitializer)
// CHECK-LE: call <4 x i32> asm "vsum2sws $0,$1,$2", "=v,v,v"(<4 x i32> %11, <4 x i32> zeroinitializer)
// CHECK-BE: call <4 x i32> @vec_sum2s(<4 x i32> noundef %{{[0-9a-zA-Z_.]+}}, <4 x i32> noundef zeroinitializer)
// CHECK-BE: call <4 x i32> @vec_sld(int vector[4], int vector[4], unsigned int)

void __attribute__((noinline))
test_set() {
  resi = _mm_set_epi16(ss[7], ss[6], ss[5], ss[4], ss[3], ss[2], ss[1], ss[0]);
  resi = _mm_set_epi32(is[3], is[2], is[1], is[0]);
  resi = _mm_set_epi64(m641, m642);
  resi = _mm_set_epi64x(i64s[0], i64s[1]);
  resi = _mm_set_epi8(chs[15], chs[14], chs[13], chs[12], chs[11], chs[10], chs[9], chs[8], chs[7], chs[6], chs[5], chs[4], chs[3], chs[2], chs[1], chs[0]);
  resd = _mm_set_pd(dp[0], dp[1]);
  resd = _mm_set_pd1(dp[0]);
  resd = _mm_set_sd(dp[0]);
  resi = _mm_set1_epi16(ss[0]);
  resi = _mm_set1_epi32(i);
  resi = _mm_set1_epi64(m641);
  resi = _mm_set1_epi64x(i64s[0]);
  resi = _mm_set1_epi8(chs[0]);
  resd = _mm_set1_pd(dp[0]);
  resi = _mm_setr_epi16(ss[7], ss[6], ss[5], ss[4], ss[3], ss[2], ss[1], ss[0]);
  resi = _mm_setr_epi32(is[3], is[2], is[1], is[0]);
  resi = _mm_setr_epi64(m641, m642);
  resi = _mm_setr_epi8(chs[15], chs[14], chs[13], chs[12], chs[11], chs[10], chs[9], chs[8], chs[7], chs[6], chs[5], chs[4], chs[3], chs[2], chs[1], chs[0]);
  resd = _mm_setr_pd(dp[0], dp[1]);
  resd = _mm_setzero_pd();
  resi = _mm_setzero_si128();
}

// CHECK-LABEL: @test_set

// CHECK-LABEL: define available_externally <2 x i64> @_mm_set_epi16
// CHECK-COUNT-8: store i16 {{[0-9a-zA-Z_%.]+}}, i16* {{[0-9a-zA-Z_%.]+}}, align 2
// CHECK: insertelement <8 x i16> undef, i16 {{[0-9a-zA-Z_%.]+}}, i32 0
// CHECK-COUNT-7: insertelement <8 x i16> {{[0-9a-zA-Z_%.]+}}, i16 {{[0-9a-zA-Z_%.]+}}, i32 {{[1-7]}}

// CHECK-LABEL: define available_externally <2 x i64> @_mm_set_epi32
// CHECK-COUNT-4: store i32 {{[0-9a-zA-Z_%.]+}}, i32* {{[0-9a-zA-Z_%.]+}}, align 4
// CHECK: insertelement <4 x i32> undef, i32 {{[0-9a-zA-Z_%.]+}}, i32 0
// CHECK-COUNT-3: insertelement <4 x i32> {{[0-9a-zA-Z_%.]+}}, i32 {{[0-9a-zA-Z_%.]+}}, i32 {{[1-3]}}

// CHECK-LABEL: define available_externally <2 x i64> @_mm_set_epi64
// CHECK: call <2 x i64> @_mm_set_epi64x(i64 noundef %{{[0-9a-zA-Z_.]+}}, i64 noundef %{{[0-9a-zA-Z_.]+}})

// CHECK-LABEL: define available_externally <2 x i64> @_mm_set_epi64x
// CHECK: %[[VEC:[0-9a-zA-Z_.]+]] = insertelement <2 x i64> undef, i64 %{{[0-9a-zA-Z_.]+}}, i32 0
// CHECK: insertelement <2 x i64> %[[VEC]], i64 %{{[0-9a-zA-Z_.]+}}, i32 1

// CHECK-LABEL: define available_externally <2 x i64> @_mm_set_epi8
// CHECK-COUNT-16: store i8 {{[0-9a-zA-Z_%.]+}}, i8* {{[0-9a-zA-Z_%.]+}}, align 1
// CHECK: insertelement <16 x i8> undef, i8 {{[0-9a-zA-Z_%.]+}}, i32 {{[0-9]+}}
// CHECK-COUNT-15: {{[0-9a-zA-Z_%.]+}} = insertelement <16 x i8> {{[0-9a-zA-Z_%.]+}}, i8 {{[0-9a-zA-Z_%.]+}}, i32 {{[0-9]+}}

// CHECK-LABEL: define available_externally <2 x double> @_mm_set_pd
// CHECK: %[[VEC:[0-9a-zA-Z_.]+]] = insertelement <2 x double> undef, double %{{[0-9a-zA-Z_.]+}}, i32 0
// CHECK: insertelement <2 x double> %[[VEC]], double %{{[0-9a-zA-Z_.]+}}, i32 1

// CHECK-LABEL: define available_externally <2 x double> @_mm_set_pd1
// CHECK: call <2 x double> @_mm_set1_pd(double noundef %{{[0-9a-zA-Z_.]+}})

// CHECK-LABEL: define available_externally <2 x double> @_mm_set_sd
// CHECK: %[[VEC:[0-9a-zA-Z_.]+]] = insertelement <2 x double> undef, double %{{[0-9a-zA-Z_.]+}}, i32 0
// CHECK: insertelement <2 x double> %[[VEC]], double 0.000000e+00, i32 1

// CHECK-LABEL: define available_externally <2 x i64> @_mm_set1_epi16
// CHECK-COUNT-8: load i16, i16* %{{[0-9a-zA-Z_.]+}}, align 2
// CHECK: call <2 x i64> @_mm_set_epi16

// CHECK-LABEL: define available_externally <2 x i64> @_mm_set1_epi32
// CHECK-COUNT-4: load i32, i32* %{{[0-9a-zA-Z_.]+}}, align 4
// CHECK: call <2 x i64> @_mm_set_epi32

// CHECK-LABEL: define available_externally <2 x i64> @_mm_set1_epi64
// CHECK: %[[VAL1:[0-9a-zA-Z_.]+]] = load i64, i64* %{{[0-9a-zA-Z_.]+}}, align 8
// CHECK: %[[VAL2:[0-9a-zA-Z_.]+]] = load i64, i64* %{{[0-9a-zA-Z_.]+}}, align 8
// CHECK: call <2 x i64> @_mm_set_epi64(i64 noundef %[[VAL1]], i64 noundef %[[VAL2]])

// CHECK-LABEL: define available_externally <2 x i64> @_mm_set1_epi64x
// CHECK: %[[VAL1:[0-9a-zA-Z_.]+]] = load i64, i64* %{{[0-9a-zA-Z_.]+}}, align 8
// CHECK: %[[VAL2:[0-9a-zA-Z_.]+]] = load i64, i64* %{{[0-9a-zA-Z_.]+}}, align 8
// CHECK: call <2 x i64> @_mm_set_epi64x(i64 noundef %[[VAL1]], i64 noundef %[[VAL2]])

// CHECK-LABEL: define available_externally <2 x i64> @_mm_set1_epi8
// CHECK-COUNT-16: load i8, i8* %{{[0-9a-zA-Z_.]+}}, align 1
// CHECK: call <2 x i64> @_mm_set_epi8

// CHECK-LABEL: define available_externally <2 x double> @_mm_set1_pd
// CHECK: %[[VEC:[0-9a-zA-Z_.]+]] = insertelement <2 x double> undef, double %{{[0-9a-zA-Z_.]+}}, i32 0
// CHECK: insertelement <2 x double> %[[VEC]], double %{{[0-9a-zA-Z_.]+}}, i32 1

// CHECK-LABEL: define available_externally <2 x i64> @_mm_setr_epi16
// CHECK-COUNT-8: load i16, i16* {{[0-9a-zA-Z_%.]+}}, align 2
// CHECK: call <2 x i64> @_mm_set_epi16

// CHECK-LABEL: define available_externally <2 x i64> @_mm_setr_epi32
// CHECK-COUNT-4: load i32, i32* {{[0-9a-zA-Z_%.]+}}, align 4
// CHECK: call <2 x i64> @_mm_set_epi32

// CHECK-LABEL: define available_externally <2 x i64> @_mm_setr_epi64
// CHECK: %[[VAL1:[0-9a-zA-Z_.]+]] = load i64, i64* %{{[0-9a-zA-Z_.]+}}, align 8
// CHECK: %[[VAL2:[0-9a-zA-Z_.]+]] = load i64, i64* %{{[0-9a-zA-Z_.]+}}, align 8
// CHECK: call <2 x i64> @_mm_set_epi64(i64 noundef %[[VAL1]], i64 noundef %[[VAL2]])

// CHECK-LABEL: define available_externally <2 x i64> @_mm_setr_epi8
// CHECK-COUNT-16: load i8, i8* {{[0-9a-zA-Z_%.]+}}, align 1
// CHECK: call <2 x i64> @_mm_set_epi8

// CHECK-LABEL: define available_externally <2 x double> @_mm_setr_pd
// CHECK: %[[VEC:[0-9a-zA-Z_.]+]] = insertelement <2 x double> undef, double %{{[0-9a-zA-Z_.]+}}, i32 0
// CHECK: insertelement <2 x double> %[[VEC]], double %{{[0-9a-zA-Z_.]+}}, i32 1

// CHECK-LABEL: define available_externally <2 x double> @_mm_setzero_pd()
// CHECK: call <4 x i32> @vec_splats(int)(i32 noundef signext 0)

// CHECK-LABEL: define available_externally <2 x i64> @_mm_setzero_si128()
// CHECK: store <4 x i32> zeroinitializer, <4 x i32>* %{{[0-9a-zA-Z_.]+}}, align 16

void __attribute__((noinline))
test_shuffle() {
  resi = _mm_shuffle_epi32(mi1, i);
  resd = _mm_shuffle_pd(md1, md2, i);
  resi = _mm_shufflehi_epi16(mi1, i);
  resi = _mm_shufflelo_epi16(mi1, i);
}

// CHECK-LABEL: @test_shuffle

// CHECK-LABEL: define available_externally <2 x i64> @_mm_shuffle_epi32
// CHECK: %[[AND:[0-9a-zA-Z_.]+]] = and i32 %{{[0-9a-zA-Z_.]+}}, 3
// CHECK: sext i32 %[[AND]] to i64
// CHECK: %[[SHR:[0-9a-zA-Z_.]+]] = ashr i32 %{{[0-9a-zA-Z_.]+}}, 2
// CHECK: %[[AND2:[0-9a-zA-Z_.]+]] = and i32 %[[SHR]], 3
// CHECK: sext i32 %[[AND2]] to i64
// CHECK: %[[SHR2:[0-9a-zA-Z_.]+]] = ashr i32 %{{[0-9a-zA-Z_.]+}}, 4
// CHECK: %[[AND3:[0-9a-zA-Z_.]+]] = and i32 %[[SHR2]], 3
// CHECK: sext i32 %[[AND3]] to i64
// CHECK: %[[SHR:[0-9a-zA-Z_.]+]] = ashr i32 %{{[0-9a-zA-Z_.]+}}, 6
// CHECK: %[[AND4:[0-9a-zA-Z_.]+]] = and i32 %[[SHR]], 3
// CHECK: sext i32 %[[AND4]] to i64
// CHECK: getelementptr inbounds [4 x i32], [4 x i32]* @_mm_shuffle_epi32.__permute_selectors, i64 0, i64 %{{[0-9a-zA-Z_.]+}}
// CHECK: insertelement <4 x i32> %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 0
// CHECK: getelementptr inbounds [4 x i32], [4 x i32]* @_mm_shuffle_epi32.__permute_selectors, i64 0, i64 %{{[0-9a-zA-Z_.]+}}
// CHECK: insertelement <4 x i32> %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 1
// CHECK: getelementptr inbounds [4 x i32], [4 x i32]* @_mm_shuffle_epi32.__permute_selectors, i64 0, i64 %{{[0-9a-zA-Z_.]+}}
// CHECK: %[[ADD:[0-9a-zA-Z_.]+]] = add i32 %{{[0-9a-zA-Z_.]+}}, 269488144
// CHECK: insertelement <4 x i32> %{{[0-9a-zA-Z_.]+}}, i32 %[[ADD]], i32 2
// CHECK: getelementptr inbounds [4 x i32], [4 x i32]* @_mm_shuffle_epi32.__permute_selectors, i64 0, i64 %{{[0-9a-zA-Z_.]+}}
// CHECK: add i32 %{{[0-9a-zA-Z_.]+}}, 269488144
// CHECK: call <4 x i32> @vec_perm(int vector[4], int vector[4], unsigned char vector[16])

// CHECK-LABEL: define available_externally <2 x double> @_mm_shuffle_pd
// CHECK: and i32 %{{[0-9a-zA-Z_.]+}}, 3
// CHECK: %[[CMP:[0-9a-zA-Z_.]+]] = icmp eq i32 %{{[0-9a-zA-Z_.]+}}, 0
// CHECK: br i1 %[[CMP]]
// CHECK: call <2 x double> @vec_mergeh(double vector[2], double vector[2])
// CHECK: %[[CMP2:[0-9a-zA-Z_.]+]] = icmp eq i32 %{{[0-9a-zA-Z_.]+}}, 1
// CHECK: br i1 %[[CMP2]]
// CHECK: shufflevector <2 x i64> %{{[0-9a-zA-Z_.]+}}, <2 x i64> %{{[0-9a-zA-Z_.]+}}, <2 x i32> <i32 1, i32 2>
// CHECK: %[[CMP3:[0-9a-zA-Z_.]+]] = icmp eq i32 %{{[0-9a-zA-Z_.]+}}, 2
// CHECK: br i1 %[[CMP3]]
// CHECK: shufflevector <2 x i64> %{{[0-9a-zA-Z_.]+}}, <2 x i64> %{{[0-9a-zA-Z_.]+}}, <2 x i32> <i32 0, i32 3>
// CHECK: call <2 x double> @vec_mergel(double vector[2], double vector[2])

// CHECK-LABEL: define available_externally <2 x i64> @_mm_shufflehi_epi16
// CHECK: %[[AND:[0-9a-zA-Z_.]+]] = and i32 %{{[0-9a-zA-Z_.]+}}, 3
// CHECK: sext i32 %[[AND]] to i64
// CHECK: %[[SHR:[0-9a-zA-Z_.]+]] = ashr i32 %{{[0-9a-zA-Z_.]+}}, 2
// CHECK: %[[AND2:[0-9a-zA-Z_.]+]] = and i32 %[[SHR]], 3
// CHECK: sext i32 %[[AND2]] to i64
// CHECK: %[[SHR2:[0-9a-zA-Z_.]+]] = ashr i32 %{{[0-9a-zA-Z_.]+}}, 4
// CHECK: %[[AND3:[0-9a-zA-Z_.]+]] = and i32 %[[SHR2]], 3
// CHECK: sext i32 %[[AND3]] to i64
// CHECK: %[[SHR3:[0-9a-zA-Z_.]+]] = ashr i32 %{{[0-9a-zA-Z_.]+}}, 6
// CHECK: %[[AND4:[0-9a-zA-Z_.]+]] = and i32 %[[SHR3]], 3
// CHECK: sext i32 %[[AND4]] to i64
// CHECK-LE: store <2 x i64> <i64 1663540288323457296, i64 0>, <2 x i64>* %{{[0-9a-zA-Z_.]+}}, align 16
// CHECK-BE: store <2 x i64> <i64 1157726452361532951, i64 0>, <2 x i64>* %{{[0-9a-zA-Z_.]+}}, align 16
// CHECK-COUNT-4: getelementptr inbounds [4 x i16], [4 x i16]* @_mm_shufflehi_epi16.__permute_selectors, i64 0, i64 {{[0-9a-zA-Z_%.]+}}
// CHECK: call <2 x i64> @vec_perm(unsigned long long vector[2], unsigned long long vector[2], unsigned char vector[16])

// CHECK-LABEL: define available_externally <2 x i64> @_mm_shufflelo_epi16
// CHECK: %[[AND:[0-9a-zA-Z_.]+]] = and i32 {{[0-9a-zA-Z_%.]+}}, 3
// CHECK: sext i32 %[[AND]] to i64
// CHECK: %[[SHR:[0-9a-zA-Z_.]+]] = ashr i32 {{[0-9a-zA-Z_%.]+}}, 2
// CHECK: %[[AND2:[0-9a-zA-Z_.]+]] = and i32 %[[SHR]], 3
// CHECK: sext i32 %[[AND2]] to i64
// CHECK: %[[SHR2:[0-9a-zA-Z_.]+]] = ashr i32 {{[0-9a-zA-Z_%.]+}}, 4
// CHECK: %[[AND3:[0-9a-zA-Z_.]+]] = and i32 %[[SHR2]], 3
// CHECK: sext i32 %[[AND3]] to i64
// CHECK: %[[SHR3:[0-9a-zA-Z_.]+]] = ashr i32 {{[0-9a-zA-Z_%.]+}}, 6
// CHECK: %[[AND4:[0-9a-zA-Z_.]+]] = and i32 %[[SHR3]], 3
// CHECK: sext i32 %[[AND4]] to i64
// CHECK-LE: store <2 x i64> <i64 0, i64 2242261671028070680>, <2 x i64>* %{{[0-9a-zA-Z_.]+}}, align 16
// CHECK-BE: store <2 x i64> <i64 0, i64 1736447835066146335>, <2 x i64>* %{{[0-9a-zA-Z_.]+}}, align 16
// CHECK-COUNT-4: getelementptr inbounds [4 x i16], [4 x i16]* @_mm_shufflelo_epi16.__permute_selectors, i64 0, i64 {{[0-9a-zA-Z_%.]+}}
// CHECK: call <2 x i64> @vec_perm(unsigned long long vector[2], unsigned long long vector[2], unsigned char vector[16])

void __attribute__((noinline))
test_sll() {
  resi = _mm_sll_epi16(mi1, mi2);
  resi = _mm_sll_epi32(mi1, mi2);
  resi = _mm_sll_epi64(mi1, mi2);
  resi = _mm_slli_epi16(mi1, i);
  resi = _mm_slli_epi32(mi1, i);
  resi = _mm_slli_epi64(mi1, i);
  resi = _mm_slli_si128(mi1, i);
}

// CHECK-LABEL: @test_sll

// CHECK-LABEL: define available_externally <2 x i64> @_mm_sll_epi16
// CHECK: store <8 x i16> <i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15>, <8 x i16>* %{{[0-9a-zA-Z_.]+}}, align 16
// CHECK-LE: call <8 x i16> @vec_splat(unsigned short vector[8], unsigned int)
// CHECK-BE: call <8 x i16> @vec_splat(unsigned short vector[8], unsigned int)
// CHECK: call <8 x i16> @vec_cmple(unsigned short vector[8], unsigned short vector[8])(<8 x i16> noundef %{{[0-9a-zA-Z_.]+}}, <8 x i16> noundef <i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15>)
// CHECK: call <8 x i16> @vec_sl(unsigned short vector[8], unsigned short vector[8])
// CHECK: call <8 x i16> @vec_sel(unsigned short vector[8], unsigned short vector[8], bool vector[8])

// CHECK-LABEL: define available_externally <2 x i64> @_mm_sll_epi32
// CHECK-LE: call <4 x i32> @vec_splat(unsigned int vector[4], unsigned int)(<4 x i32> noundef %{{[0-9a-zA-Z_.]+}}, i32 noundef zeroext 0)
// CHECK-BE: call <4 x i32> @vec_splat(unsigned int vector[4], unsigned int)(<4 x i32> noundef %{{[0-9a-zA-Z_.]+}}, i32 noundef zeroext 1)
// CHECK: call <4 x i32> @vec_cmplt(unsigned int vector[4], unsigned int vector[4])(<4 x i32> noundef {{[0-9a-zA-Z_%.]+}}, <4 x i32> noundef <i32 32, i32 32, i32 32, i32 32>)
// CHECK: call <4 x i32> @vec_sl(unsigned int vector[4], unsigned int vector[4])
// CHECK: call <4 x i32> @vec_sel(unsigned int vector[4], unsigned int vector[4], bool vector[4])

// CHECK-LABEL: define available_externally <2 x i64> @_mm_sll_epi64
// CHECK: call <2 x i64> @vec_splat(unsigned long long vector[2], unsigned int)(<2 x i64> noundef {{[0-9a-zA-Z_%.]+}}, i32 noundef zeroext 0)
// CHECK: call <2 x i64> @vec_cmplt(unsigned long long vector[2], unsigned long long vector[2])(<2 x i64> noundef {{[0-9a-zA-Z_%.]+}}, <2 x i64> noundef <i64 64, i64 64>)
// CHECK: call <2 x i64> @vec_sl(unsigned long long vector[2], unsigned long long vector[2])
// CHECK: call <2 x i64> @vec_sel(unsigned long long vector[2], unsigned long long vector[2], bool vector[2])

// CHECK-LABEL: define available_externally <2 x i64> @_mm_slli_epi16
// CHECK: store <8 x i16> zeroinitializer, <8 x i16>* %{{[0-9a-zA-Z_.]+}}, align 16
// CHECK: %[[CMP:[0-9a-zA-Z_.]+]] = icmp sge i32 %{{[0-9a-zA-Z_.]+}}, 0
// CHECK: br i1 %[[CMP]]
// CHECK: %[[CMP2:[0-9a-zA-Z_.]+]] = icmp slt i32 %{{[0-9a-zA-Z_.]+}}, 16
// CHECK: br i1 %[[CMP2]]
// CHECK: call i1 @llvm.is.constant
// CHECK: %[[TRUNC:[0-9a-zA-Z_.]+]] = trunc i32 %{{[0-9a-zA-Z_.]+}} to i8
// CHECK: call <8 x i16> @vec_splat_s16(signed char)(i8 noundef signext %[[TRUNC]])
// CHECK: %[[TRUNC2:[0-9a-zA-Z_.]+]] = trunc i32 %{{[0-9a-zA-Z_.]+}} to i16
// CHECK: call <8 x i16> @vec_splats(unsigned short)(i16 noundef zeroext %[[TRUNC2]])
// CHECK: call <8 x i16> @vec_sl(short vector[8], unsigned short vector[8])

// CHECK-LABEL: define available_externally <2 x i64> @_mm_slli_epi32
// CHECK: store <4 x i32> zeroinitializer, <4 x i32>* %{{[0-9a-zA-Z_.]+}}, align 16
// CHECK: %[[CMP:[0-9a-zA-Z_.]+]] = icmp sge i32 %{{[0-9a-zA-Z_.]+}}, 0
// CHECK: br i1 %[[CMP]]
// CHECK: %[[CMP2:[0-9a-zA-Z_.]+]] = icmp slt i32 %{{[0-9a-zA-Z_.]+}}, 32
// CHECK: br i1 %[[CMP2]]
// CHECK: call i1 @llvm.is.constant
// CHECK: %[[CMP3:[0-9a-zA-Z_.]+]] = icmp slt i32 %{{[0-9a-zA-Z_.]+}}, 16
// CHECK: br i1 %[[CMP3]]
// CHECK: %[[TRUNC:[0-9a-zA-Z_.]+]] = trunc i32 %{{[0-9a-zA-Z_.]+}} to i8
// CHECK: call <4 x i32> @vec_splat_s32(signed char)(i8 noundef signext %[[TRUNC]])
// CHECK: call <4 x i32> @vec_splats(unsigned int)
// CHECK: call <4 x i32> @vec_sl(int vector[4], unsigned int vector[4])

// CHECK-LABEL: define available_externally <2 x i64> @_mm_slli_epi64
// CHECK: store <2 x i64> zeroinitializer, <2 x i64>* %{{[0-9a-zA-Z_.]+}}, align 16
// CHECK: %[[CMP:[0-9a-zA-Z_.]+]] = icmp sge i32 %{{[0-9a-zA-Z_.]+}}, 0
// CHECK: br i1 %[[CMP]]
// CHECK: %[[CMP2:[0-9a-zA-Z_.]+]] = icmp slt i32 %{{[0-9a-zA-Z_.]+}}, 64
// CHECK: br i1 %[[CMP2]]
// CHECK: call i1 @llvm.is.constant
// CHECK: %[[CMP3:[0-9a-zA-Z_.]+]] = icmp slt i32 %{{[0-9a-zA-Z_.]+}}, 16
// CHECK: br i1 %[[CMP3]]
// CHECK: %[[TRUNC:[0-9a-zA-Z_.]+]] = trunc i32 %{{[0-9a-zA-Z_.]+}} to i8
// CHECK: call <4 x i32> @vec_splat_s32(signed char)(i8 noundef signext %[[TRUNC]])
// CHECK: call <4 x i32> @vec_splats(unsigned int)
// CHECK: call <2 x i64> @vec_sl(long long vector[2], unsigned long long vector[2])

// CHECK-LABEL: define available_externally <2 x i64> @_mm_slli_si128
// CHECK: store <16 x i8> zeroinitializer, <16 x i8>* %{{[0-9a-zA-Z_.]+}}, align 16
// CHECK-BE: %[[SUB:[0-9a-zA-Z_.]+]] = sub nsw i32 16, %{{[0-9a-zA-Z_.]+}}
// CHECK-BE: call <16 x i8> @vec_sld(unsigned char vector[16], unsigned char vector[16], unsigned int)(<16 x i8> noundef zeroinitializer, <16 x i8> noundef %{{[0-9a-zA-Z_.]+}}, i32 noundef zeroext %[[SUB]])
// CHECK-LE: call <16 x i8> @vec_sld(unsigned char vector[16], unsigned char vector[16], unsigned int)(<16 x i8> noundef %{{[0-9a-zA-Z_.]+}}, <16 x i8> noundef zeroinitializer, i32 noundef zeroext %{{[0-9a-zA-Z_.]+}})
// CHECK: store <16 x i8> zeroinitializer, <16 x i8>* %{{[0-9a-zA-Z_.]+}}, align 16

void __attribute__((noinline))
test_sqrt() {
  resd = _mm_sqrt_pd(md1);
  resd = _mm_sqrt_sd(md1, md2);
}

// CHECK-LABEL: @test_sqrt

// CHECK-LABEL: define available_externally <2 x double> @_mm_sqrt_pd
// CHECK: call <2 x double> @vec_sqrt(double vector[2])(<2 x double> noundef {{[0-9a-zA-Z_%.]+}})

// CHECK-LABEL: define available_externally <2 x double> @_mm_sqrt_sd
// CHECK: %[[CALL:[0-9a-zA-Z_.]+]] = call <2 x double> @_mm_set1_pd(double noundef %{{[0-9a-zA-Z_.]+}})
// CHECK: call <2 x double> @vec_sqrt(double vector[2])(<2 x double> noundef %{{[0-9a-zA-Z_.]+}})
// CHECK: call <2 x double> @_mm_setr_pd(double noundef %{{[0-9a-zA-Z_.]+}}, double noundef %{{[0-9a-zA-Z_.]+}})

void __attribute__((noinline))
test_sra() {
  resi = _mm_sra_epi16(mi1, mi2);
  resi = _mm_sra_epi32(mi1, mi2);
  resi = _mm_srai_epi16(mi1, i);
  resi = _mm_srai_epi32(mi1, i);
}

// CHECK-LABEL: @test_sra

// CHECK-LABEL: define available_externally <2 x i64> @_mm_sra_epi16
// CHECK: store <8 x i16> <i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15>, <8 x i16>* %{{[0-9a-zA-Z_.]+}}, align 16
// CHECK-LE: call <8 x i16> @vec_splat(unsigned short vector[8], unsigned int)(<8 x i16> noundef %{{[0-9a-zA-Z_.]+}}, i32 noundef zeroext 0)
// CHECK-BE: call <8 x i16> @vec_splat(unsigned short vector[8], unsigned int)(<8 x i16> noundef %{{[0-9a-zA-Z_.]+}}, i32 noundef zeroext 3)
// CHECK: call <8 x i16> @vec_min(unsigned short vector[8], unsigned short vector[8])(<8 x i16> noundef %{{[0-9a-zA-Z_.]+}}, <8 x i16> noundef <i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15>)
// CHECK: call <8 x i16> @vec_sra(short vector[8], unsigned short vector[8])

// CHECK-LABEL: define available_externally <2 x i64> @_mm_sra_epi32
// CHECK: store <4 x i32> <i32 31, i32 31, i32 31, i32 31>, <4 x i32>* %{{[0-9a-zA-Z_.]+}}, align 16
// CHECK-LE: call <4 x i32> @vec_splat(unsigned int vector[4], unsigned int)(<4 x i32> noundef %{{[0-9a-zA-Z_.]+}}, i32 noundef zeroext 0)
// CHECK-BE: call <4 x i32> @vec_splat(unsigned int vector[4], unsigned int)(<4 x i32> noundef %{{[0-9a-zA-Z_.]+}}, i32 noundef zeroext 1)
// CHECK: call <4 x i32> @vec_min(unsigned int vector[4], unsigned int vector[4])(<4 x i32> noundef %{{[0-9a-zA-Z_.]+}}, <4 x i32> noundef <i32 31, i32 31, i32 31, i32 31>)
// CHECK: call <4 x i32> @vec_sra(int vector[4], unsigned int vector[4])

// CHECK-LABEL: define available_externally <2 x i64> @_mm_srai_epi16
// CHECK: store <8 x i16> <i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15>, <8 x i16>* %{{[0-9a-zA-Z_.]+}}, align 16
// CHECK: %[[CMP:[0-9a-zA-Z_.]+]] = icmp slt i32 %{{[0-9a-zA-Z_.]+}}, 16
// CHECK: br i1 %[[CMP]]
// CHECK: call i1 @llvm.is.constant
// CHECK: %[[TRUNC:[0-9a-zA-Z_.]+]] = trunc i32 %{{[0-9a-zA-Z_.]+}} to i8
// CHECK: call <8 x i16> @vec_splat_s16(signed char)(i8 noundef signext %[[TRUNC]])
// CHECK: %[[TRUNC2:[0-9a-zA-Z_.]+]] = trunc i32 %{{[0-9a-zA-Z_.]+}} to i16
// CHECK: call <8 x i16> @vec_splats(unsigned short)(i16 noundef zeroext %{{[0-9a-zA-Z_.]+}})
// CHECK: call <8 x i16> @vec_sra(short vector[8], unsigned short vector[8])

// CHECK-LABEL: define available_externally <2 x i64> @_mm_srai_epi32
// CHECK: store <4 x i32> <i32 31, i32 31, i32 31, i32 31>, <4 x i32>* %{{[0-9a-zA-Z_.]+}}, align 16
// CHECK: %[[CMP:[0-9a-zA-Z_.]+]] = icmp slt i32 %{{[0-9a-zA-Z_.]+}}, 32
// CHECK: br i1 %[[CMP]]
// CHECK: call i1 @llvm.is.constant
// CHECK: %[[CMP2:[0-9a-zA-Z_.]+]] = icmp slt i32 %{{[0-9a-zA-Z_.]+}}, 16
// CHECK: br i1 %[[CMP2]]
// CHECK: %[[TRUNC:[0-9a-zA-Z_.]+]] = trunc i32 %{{[0-9a-zA-Z_.]+}} to i8
// CHECK: call <4 x i32> @vec_splat_s32(signed char)(i8 noundef signext %[[TRUNC]])
// CHECK: call <4 x i32> @vec_splats(unsigned int)
// CHECK: call <4 x i32> @vec_splats(unsigned int)
// CHECK: call <4 x i32> @vec_sra(int vector[4], unsigned int vector[4])

void __attribute__((noinline))
test_srl() {
  resi = _mm_srl_epi16(mi1, mi2);
  resi = _mm_srl_epi32(mi1, mi2);
  resi = _mm_srl_epi64(mi1, mi2);
  resi = _mm_srli_epi16(mi1, i);
  resi = _mm_srli_epi32(mi1, i);
  resi = _mm_srli_epi64(mi1, i);
  resi = _mm_srli_si128(mi1, i);
}

// CHECK-LABEL: @test_srl

// CHECK-LABEL: define available_externally <2 x i64> @_mm_srl_epi16
// CHECK: store <8 x i16> <i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15>, <8 x i16>* %{{[0-9a-zA-Z_.]+}}, align 16
// CHECK-LE: call <8 x i16> @vec_splat(unsigned short vector[8], unsigned int)(<8 x i16> noundef %{{[0-9a-zA-Z_.]+}}, i32 noundef zeroext 0)
// CHECK-BE: call <8 x i16> @vec_splat(unsigned short vector[8], unsigned int)(<8 x i16> noundef %{{[0-9a-zA-Z_.]+}}, i32 noundef zeroext 3)
// CHECK: call <8 x i16> @vec_cmple(unsigned short vector[8], unsigned short vector[8])(<8 x i16> noundef %{{[0-9a-zA-Z_.]+}}, <8 x i16> noundef <i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15>)
// CHECK: call <8 x i16> @vec_sr(unsigned short vector[8], unsigned short vector[8])
// CHECK: call <8 x i16> @vec_sel(unsigned short vector[8], unsigned short vector[8], bool vector[8])

// CHECK-LABEL: define available_externally <2 x i64> @_mm_srl_epi32
// CHECK-LE: call <4 x i32> @vec_splat(unsigned int vector[4], unsigned int)(<4 x i32> noundef %{{[0-9a-zA-Z_.]+}}, i32 noundef zeroext 0)
// CHECK-BE: call <4 x i32> @vec_splat(unsigned int vector[4], unsigned int)(<4 x i32> noundef %{{[0-9a-zA-Z_.]+}}, i32 noundef zeroext 1)
// CHECK: call <4 x i32> @vec_cmplt(unsigned int vector[4], unsigned int vector[4])(<4 x i32> noundef %{{[0-9a-zA-Z_.]+}}, <4 x i32> noundef <i32 32, i32 32, i32 32, i32 32>)
// CHECK: call <4 x i32> @vec_sr(unsigned int vector[4], unsigned int vector[4])
// CHECK: call <4 x i32> @vec_sel(unsigned int vector[4], unsigned int vector[4], bool vector[4])

// CHECK-LABEL: define available_externally <2 x i64> @_mm_srl_epi64
// CHECK: call <2 x i64> @vec_splat(unsigned long long vector[2], unsigned int)(<2 x i64> noundef %{{[0-9a-zA-Z_.]+}}, i32 noundef zeroext 0)
// CHECK: call <2 x i64> @vec_cmplt(unsigned long long vector[2], unsigned long long vector[2])(<2 x i64> noundef %{{[0-9a-zA-Z_.]+}}, <2 x i64> noundef <i64 64, i64 64>)
// CHECK: call <2 x i64> @vec_sr(unsigned long long vector[2], unsigned long long vector[2])
// CHECK: call <2 x i64> @vec_sel(unsigned long long vector[2], unsigned long long vector[2], bool vector[2])

// CHECK-LABEL: define available_externally <2 x i64> @_mm_srli_epi16
// CHECK: store <8 x i16> zeroinitializer, <8 x i16>* %{{[0-9a-zA-Z_.]+}}, align 16
// CHECK: %[[CMP:[0-9a-zA-Z_.]+]] = icmp slt i32 %{{[0-9a-zA-Z_.]+}}, 16
// CHECK: br i1 %[[CMP]]
// CHECK: call i1 @llvm.is.constant
// CHECK: trunc i32 %{{[0-9a-zA-Z_.]+}} to i8
// CHECK: call <8 x i16> @vec_splat_s16(signed char)
// CHECK: %[[TRUNC:[0-9a-zA-Z_.]+]] = trunc i32 %{{[0-9a-zA-Z_.]+}} to i16
// CHECK: call <8 x i16> @vec_splats(unsigned short)(i16 noundef zeroext %[[TRUNC]])
// CHECK: call <8 x i16> @vec_sr(short vector[8], unsigned short vector[8])

// CHECK-LABEL: define available_externally <2 x i64> @_mm_srli_epi32
// CHECK: store <4 x i32> zeroinitializer, <4 x i32>* %{{[0-9a-zA-Z_.]+}}, align 16
// CHECK: %[[CMP:[0-9a-zA-Z_.]+]] = icmp slt i32 %{{[0-9a-zA-Z_.]+}}, 32
// CHECK: br i1 %[[CMP]]
// CHECK: call i1 @llvm.is.constant
// CHECK: %[[CMP2:[0-9a-zA-Z_.]+]] = icmp slt i32 %{{[0-9a-zA-Z_.]+}}, 16
// CHECK: br i1 %[[CMP2]]
// CHECK: %[[TRUNC:[0-9a-zA-Z_.]+]] = trunc i32 %{{[0-9a-zA-Z_.]+}} to i8
// CHECK: call <4 x i32> @vec_splat_s32(signed char)
// CHECK: call <4 x i32> @vec_splats(unsigned int)
// CHECK: call <4 x i32> @vec_splats(unsigned int)
// CHECK: call <4 x i32> @vec_sr(int vector[4], unsigned int vector[4])

// CHECK-LABEL: define available_externally <2 x i64> @_mm_srli_epi64
// CHECK: store <2 x i64> zeroinitializer, <2 x i64>* %{{[0-9a-zA-Z_.]+}}, align 16
// CHECK: %[[CMP:[0-9a-zA-Z_.]+]] = icmp slt i32 %{{[0-9a-zA-Z_.]+}}, 64
// CHECK: br i1 %[[CMP]]
// CHECK: call i1 @llvm.is.constant
// CHECK: %[[CMP2:[0-9a-zA-Z_.]+]] = icmp slt i32 %{{[0-9a-zA-Z_.]+}}, 16
// CHECK: br i1 %[[CMP2]]
// CHECK: %[[TRUNC:[0-9a-zA-Z_.]+]] = trunc i32 %{{[0-9a-zA-Z_.]+}} to i8
// CHECK: call <4 x i32> @vec_splat_s32(signed char)(i8 noundef signext %[[TRUNC]])
// CHECK: %[[EXT:[0-9a-zA-Z_.]+]] = sext i32 %{{[0-9a-zA-Z_.]+}} to i64
// CHECK: call <2 x i64> @vec_splats(unsigned long long)(i64 noundef %[[EXT]])
// CHECK: call <4 x i32> @vec_splats(unsigned int)(i32 noundef zeroext %{{[0-9a-zA-Z_.]+}})
// CHECK: call <2 x i64> @vec_sr(long long vector[2], unsigned long long vector[2])

// CHECK-LABEL: define available_externally <2 x i64> @_mm_srli_si128
// CHECK: call <2 x i64> @_mm_bsrli_si128

void __attribute__((noinline))
test_store() {
  _mm_store_pd(dp, md1);
  _mm_store_pd1(dp, md1);
  _mm_store_sd(dp, md1);
  _mm_store_si128(mip, mi1);
  _mm_store1_pd(dp, md1);
  _mm_storeh_pd(dp, md1);
  _mm_storel_epi64(mip, mi1);
  _mm_storel_pd(dp, md1);
  _mm_storer_pd(dp, md1);
  _mm_storeu_pd(dp, md1);
  _mm_storeu_si128(mip, mi1);
}

// CHECK-LABEL: @test_store

// CHECK-LABEL: define available_externally void @_mm_store_pd
// CHECK: %[[ADDR:[0-9a-zA-Z_.]+]] = load double*, double** %{{[0-9a-zA-Z_.]+}}, align 8
// CHECK: %[[CAST:[0-9a-zA-Z_.]+]] = bitcast double* %[[ADDR]] to <16 x i8>*
// CHECK: call void @vec_st(unsigned char vector[16], long, unsigned char vector[16]*)(<16 x i8> noundef %{{[0-9a-zA-Z_.]+}}, i64 noundef 0, <16 x i8>* noundef %[[CAST]])

// CHECK-LABEL: define available_externally void @_mm_store_pd1
// CHECK: %[[ADDR:[0-9a-zA-Z_.]+]] = load double*, double** %{{[0-9a-zA-Z_.]+}}, align 8
// CHECK: %[[ADDR2:[0-9a-zA-Z_.]+]] = load <2 x double>, <2 x double>* %{{[0-9a-zA-Z_.]+}}, align 16
// CHECK: call void @_mm_store1_pd(double* noundef %[[ADDR]], <2 x double> noundef %[[ADDR2]])

// CHECK-LABEL: define available_externally void @_mm_store_sd
// CHECK: %[[ADDR:[0-9a-zA-Z_.]+]] = load double*, double** %{{[0-9a-zA-Z_.]+}}, align 8
// CHECK: store double %{{[0-9a-zA-Z_.]+}}, double* %[[ADDR]], align 8

// CHECK-LABEL: define available_externally void @_mm_store_si128
// CHECK: %[[LOAD:[0-9a-zA-Z_.]+]] = load <2 x i64>*, <2 x i64>** %{{[0-9a-zA-Z_.]+}}, align 8
// CHECK: %[[CAST:[0-9a-zA-Z_.]+]] = bitcast <2 x i64>* %[[LOAD]] to <16 x i8>*
// CHECK: call void @vec_st(unsigned char vector[16], long, unsigned char vector[16]*)(<16 x i8> noundef %{{[0-9a-zA-Z_.]+}}, i64 noundef 0, <16 x i8>* noundef %[[CAST]])

// CHECK-LABEL: define available_externally void @_mm_store1_pd
// CHECK: %[[ADDR:[0-9a-zA-Z_.]+]] = load double*, double** %{{[0-9a-zA-Z_.]+}}, align 8
// CHECK: %[[VAL:[0-9a-zA-Z_.]+]] = load <2 x double>, <2 x double>* %{{[0-9a-zA-Z_.]+}}, align 16
// CHECK: %[[CALL:[0-9a-zA-Z_.]+]] = call <2 x double> @vec_splat(double vector[2], unsigned int)(<2 x double> noundef %[[VAL]], i32 noundef zeroext 0)
// CHECK: call void @_mm_store_pd(double* noundef %[[ADDR]], <2 x double> noundef %[[CALL]])

// CHECK-LABEL: define available_externally void @_mm_storeh_pd
// CHECK: %[[ADDR:[0-9a-zA-Z_.]+]] = load double*, double** %{{[0-9a-zA-Z_.]+}}, align 8
// CHECK: store double %{{[0-9a-zA-Z_.]+}}, double* %[[ADDR]], align 8

// CHECK-LABEL: define available_externally void @_mm_storel_epi64
// CHECK: %[[ADDR:[0-9a-zA-Z_.]+]] = load <2 x i64>*, <2 x i64>** %{{[0-9a-zA-Z_.]+}}, align 8
// CHECK: %[[CAST:[0-9a-zA-Z_.]+]] = bitcast <2 x i64>* %[[ADDR]] to i64*
// CHECK: store i64 %{{[0-9a-zA-Z_.]+}}, i64* %[[CAST]], align 8

// CHECK-LABEL: define available_externally void @_mm_storel_pd
// CHECK: %[[ADDR:[0-9a-zA-Z_.]+]] = load double*, double** %{{[0-9a-zA-Z_.]+}}, align 8
// CHECK: %[[VAL:[0-9a-zA-Z_.]+]] = load <2 x double>, <2 x double>* %{{[0-9a-zA-Z_.]+}}, align 16
// CHECK: call void @_mm_store_sd(double* noundef %[[ADDR]], <2 x double> noundef %[[VAL]])

// CHECK-LABEL: define available_externally void @_mm_storer_pd
// CHECK: shufflevector <2 x i64> %{{[0-9a-zA-Z_.]+}}, <2 x i64> %{{[0-9a-zA-Z_.]+}}, <2 x i32> <i32 1, i32 2>
// CHECK: call void @_mm_store_pd(double* noundef %{{[0-9a-zA-Z_.]+}}, <2 x double> noundef %{{[0-9a-zA-Z_.]+}})

// CHECK-LABEL: define available_externally void @_mm_storeu_pd
// CHECK: %[[ADDR:[0-9a-zA-Z_.]+]] = load double*, double** %{{[0-9a-zA-Z_.]+}}, align 8
// CHECK: %[[CAST:[0-9a-zA-Z_.]+]] = bitcast double* %[[ADDR]] to <2 x double>*
// CHECK: store <2 x double> %{{[0-9a-zA-Z_.]+}}, <2 x double>* %[[CAST]], align 1

// CHECK-LABEL: define available_externally void @_mm_storeu_si128
// CHECK: %[[ADDR:[0-9a-zA-Z_.]+]] = load <2 x i64>*, <2 x i64>** %{{[0-9a-zA-Z_.]+}}, align 8
// CHECK: store <2 x i64> %{{[0-9a-zA-Z_.]+}}, <2 x i64>* %[[ADDR]], align 1

void __attribute__((noinline))
test_stream() {
  _mm_stream_pd(dp, md1);
  _mm_stream_si128(mip, mi1);
  _mm_stream_si32(is, i);
  _mm_stream_si64(i64s, i64s[1]);
}

// CHECK-LABEL: @test_stream

// CHECK-LABEL: define available_externally void @_mm_stream_pd
// CHECK: call void asm sideeffect "dcbtstt 0,$0", "b,~{memory}"(double* %{{[0-9a-zA-Z_.]+}})

// CHECK-LABEL: define available_externally void @_mm_stream_si128
// CHECK: call void asm sideeffect "dcbtstt 0,$0", "b,~{memory}"(<2 x i64>* %{{[0-9a-zA-Z_.]+}})

// CHECK-LABEL: define available_externally void @_mm_stream_si32
// CHECK: call void asm sideeffect "dcbtstt 0,$0", "b,~{memory}"(i32* %{{[0-9a-zA-Z_.]+}})

// CHECK-LABEL: define available_externally void @_mm_stream_si64
// CHECK: call void asm sideeffect "\09dcbtstt\090,$0", "b,~{memory}"(i64* %{{[0-9a-zA-Z_.]+}})

void __attribute__((noinline))
test_sub() {
  resi = _mm_sub_epi64(mi1, mi2);
  resi = _mm_sub_epi32(mi1, mi2);
  resi = _mm_sub_epi16(mi1, mi2);
  resi = _mm_sub_epi8(mi1, mi2);
  resd = _mm_sub_pd(md1, md2);
  resd = _mm_sub_sd(md1, md2);
  res64 = _mm_sub_si64(m641, m642);
  resi = _mm_subs_epi16(mi1, mi2);
  resi = _mm_subs_epi8(mi1, mi2);
  resi = _mm_subs_epu16(mi1, mi2);
  resi = _mm_subs_epu8(mi1, mi2);
}

// CHECK-LABEL: @test_sub

// CHECK-LABEL: define available_externally <2 x i64> @_mm_sub_epi64
// CHECK: sub <2 x i64>

// CHECK-LABEL: define available_externally <2 x i64> @_mm_sub_epi32
// CHECK: sub <4 x i32>

// CHECK-LABEL: define available_externally <2 x i64> @_mm_sub_epi16
// CHECK: sub <8 x i16>

// CHECK-LABEL: define available_externally <2 x i64> @_mm_sub_epi8
// CHECK: sub <16 x i8>

// CHECK-LABEL: define available_externally <2 x double> @_mm_sub_pd
// CHECK: fsub <2 x double>

// CHECK-LABEL: define available_externally <2 x double> @_mm_sub_sd
// CHECK: fsub double

// CHECK-LABEL: define available_externally i64 @_mm_sub_si64
// CHECK: sub i64

// CHECK-LABEL: define available_externally <2 x i64> @_mm_subs_epi16
// CHECK: call <8 x i16> @vec_subs(short vector[8], short vector[8])

// CHECK-LABEL: define available_externally <2 x i64> @_mm_subs_epi8
// CHECK: call <16 x i8> @vec_subs(signed char vector[16], signed char vector[16])

// CHECK-LABEL: define available_externally <2 x i64> @_mm_subs_epu16
// CHECK: call <8 x i16> @vec_subs(unsigned short vector[8], unsigned short vector[8])

// CHECK-LABEL: define available_externally <2 x i64> @_mm_subs_epu8
// CHECK: call <16 x i8> @vec_subs(unsigned char vector[16], unsigned char vector[16])

void __attribute__((noinline))
test_ucomi() {
  i = _mm_ucomieq_sd(md1, md2);
  i = _mm_ucomige_sd(md1, md2);
  i = _mm_ucomigt_sd(md1, md2);
  i = _mm_ucomile_sd(md1, md2);
  i = _mm_ucomilt_sd(md1, md2);
  i = _mm_ucomineq_sd(md1, md2);
}

// CHECK-LABEL: @test_ucomi

// CHECK-LABEL: define available_externally signext i32 @_mm_ucomieq_sd
// CHECK: fcmp oeq double

// CHECK-LABEL: define available_externally signext i32 @_mm_ucomige_sd
// CHECK: fcmp oge double

// CHECK-LABEL: define available_externally signext i32 @_mm_ucomigt_sd
// CHECK: fcmp ogt double

// CHECK-LABEL: define available_externally signext i32 @_mm_ucomile_sd
// CHECK: fcmp ole double

// CHECK-LABEL: define available_externally signext i32 @_mm_ucomilt_sd
// CHECK: fcmp olt double

// CHECK-LABEL: define available_externally signext i32 @_mm_ucomineq_sd
// CHECK: fcmp une double

void __attribute__((noinline))
test_undefined() {
  resd = _mm_undefined_pd();
  resi = _mm_undefined_si128();
}

// CHECK-LABEL: @test_undefined

// CHECK-LABEL: define available_externally <2 x double> @_mm_undefined_pd()
// CHECK: %[[VAL:[0-9a-zA-Z_.]+]] = load <2 x double>, <2 x double>* %[[ADDR:[0-9a-zA-Z_.]+]], align 16
// CHECK: store <2 x double> %[[VAL]], <2 x double>* %[[ADDR]], align 16
// CHECK: load <2 x double>, <2 x double>* %[[ADDR]], align 16

// CHECK-LABEL: define available_externally <2 x i64> @_mm_undefined_si128()
// CHECK: %[[VAL:[0-9a-zA-Z_.]+]] = load <2 x i64>, <2 x i64>* %[[ADDR:[0-9a-zA-Z_.]+]], align 16
// CHECK: store <2 x i64> %[[VAL]], <2 x i64>* %[[ADDR]], align 16
// CHECK: load <2 x i64>, <2 x i64>* %[[ADDR]], align 16

void __attribute__((noinline))
test_unpack() {
  resi = _mm_unpackhi_epi16(mi1, mi2);
  resi = _mm_unpackhi_epi32(mi1, mi2);
  resi = _mm_unpackhi_epi64(mi1, mi2);
  resi = _mm_unpackhi_epi8(mi1, mi2);
  resd = _mm_unpackhi_pd(md1, md2);
  resi = _mm_unpacklo_epi16(mi1, mi2);
  resi = _mm_unpacklo_epi32(mi1, mi2);
  resi = _mm_unpacklo_epi64(mi1, mi2);
  resi = _mm_unpacklo_epi8(mi1, mi2);
  resd = _mm_unpacklo_pd(md1, md2);
}

// CHECK-LABEL: @test_unpack

// CHECK-LABEL: define available_externally <2 x i64> @_mm_unpackhi_epi16
// CHECK: call <8 x i16> @vec_mergel(unsigned short vector[8], unsigned short vector[8])

// CHECK-LABEL: define available_externally <2 x i64> @_mm_unpackhi_epi32
// CHECK: call <4 x i32> @vec_mergel(unsigned int vector[4], unsigned int vector[4])

// CHECK-LABEL: define available_externally <2 x i64> @_mm_unpackhi_epi64
// CHECK: call <2 x i64> @vec_mergel(long long vector[2], long long vector[2])

// CHECK-LABEL: define available_externally <2 x i64> @_mm_unpackhi_epi8
// CHECK: call <16 x i8> @vec_mergel(unsigned char vector[16], unsigned char vector[16])

// CHECK-LABEL: define available_externally <2 x double> @_mm_unpackhi_pd
// CHECK: call <2 x double> @vec_mergel(double vector[2], double vector[2])

// CHECK-LABEL: define available_externally <2 x i64> @_mm_unpacklo_epi16
// CHECK: call <8 x i16> @vec_mergeh(short vector[8], short vector[8])

// CHECK-LABEL: define available_externally <2 x i64> @_mm_unpacklo_epi32
// CHECK: call <4 x i32> @vec_mergeh(int vector[4], int vector[4])

// CHECK-LABEL: define available_externally <2 x i64> @_mm_unpacklo_epi64
// CHECK: call <2 x i64> @vec_mergeh(long long vector[2], long long vector[2])

// CHECK-LABEL: define available_externally <2 x i64> @_mm_unpacklo_epi8
// CHECK: call <16 x i8> @vec_mergeh(unsigned char vector[16], unsigned char vector[16])

// CHECK-LABEL: define available_externally <2 x double> @_mm_unpacklo_pd
// CHECK: call <2 x double> @vec_mergeh(double vector[2], double vector[2])
