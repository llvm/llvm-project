ifeq ($(subdir),math)
libm-sysdep_routines += s_floor-c s_ceil-c s_floorf-c s_ceilf-c \
			s_rint-c s_rintf-c s_nearbyint-c s_nearbyintf-c \
			s_roundeven-c s_roundevenf-c s_trunc-c s_truncf-c

libm-sysdep_routines += s_ceil-sse4_1 s_ceilf-sse4_1 s_floor-sse4_1 \
			s_floorf-sse4_1 s_nearbyint-sse4_1 \
			s_nearbyintf-sse4_1 s_roundeven-sse4_1 \
			s_roundevenf-sse4_1 s_rint-sse4_1 s_rintf-sse4_1 \
			s_trunc-sse4_1 s_truncf-sse4_1

libm-sysdep_routines += e_exp-fma e_log-fma e_pow-fma s_atan-fma \
			e_asin-fma e_atan2-fma s_sin-fma s_tan-fma

CFLAGS-e_asin-fma.c = -mfma -mavx2
CFLAGS-e_atan2-fma.c = -mfma -mavx2
CFLAGS-e_exp-fma.c = -mfma -mavx2
CFLAGS-e_log-fma.c = -mfma -mavx2
CFLAGS-e_pow-fma.c = -mfma -mavx2
CFLAGS-s_atan-fma.c = -mfma -mavx2
CFLAGS-s_sin-fma.c = -mfma -mavx2
CFLAGS-s_tan-fma.c = -mfma -mavx2

libm-sysdep_routines += s_sinf-sse2 s_cosf-sse2 s_sincosf-sse2

libm-sysdep_routines += e_exp2f-fma e_expf-fma e_log2f-fma e_logf-fma \
			e_powf-fma s_sinf-fma s_cosf-fma s_sincosf-fma

CFLAGS-e_exp2f-fma.c = -mfma -mavx2
CFLAGS-e_expf-fma.c = -mfma -mavx2
CFLAGS-e_log2f-fma.c = -mfma -mavx2
CFLAGS-e_logf-fma.c = -mfma -mavx2
CFLAGS-e_powf-fma.c = -mfma -mavx2
CFLAGS-s_sinf-fma.c = -mfma -mavx2
CFLAGS-s_cosf-fma.c = -mfma -mavx2
CFLAGS-s_sincosf-fma.c = -mfma -mavx2

libm-sysdep_routines += e_exp-fma4 e_log-fma4 e_pow-fma4 s_atan-fma4 \
			e_asin-fma4 e_atan2-fma4 s_sin-fma4 s_tan-fma4

CFLAGS-e_asin-fma4.c = -mfma4
CFLAGS-e_atan2-fma4.c = -mfma4
CFLAGS-e_exp-fma4.c = -mfma4
CFLAGS-e_log-fma4.c = -mfma4
CFLAGS-e_pow-fma4.c = -mfma4
CFLAGS-s_atan-fma4.c = -mfma4
CFLAGS-s_sin-fma4.c = -mfma4
CFLAGS-s_tan-fma4.c = -mfma4

libm-sysdep_routines += e_exp-avx e_log-avx s_atan-avx \
			e_atan2-avx s_sin-avx s_tan-avx

ifeq ($(with-clang),no)
CFLAGS-e_atan2-avx.c = -msse2avx -DSSE2AVX
CFLAGS-e_exp-avx.c = -msse2avx -DSSE2AVX
CFLAGS-e_log-avx.c = -msse2avx -DSSE2AVX
CFLAGS-s_atan-avx.c = -msse2avx -DSSE2AVX
CFLAGS-s_sin-avx.c = -msse2avx -DSSE2AVX
CFLAGS-s_tan-avx.c = -msse2avx -DSSE2AVX
else
CFLAGS-e_atan2-avx.c = -msse2 -mavx2 -DSSE2AVX
CFLAGS-e_exp-avx.c = -msse2 -mavx2 -DSSE2AVX
CFLAGS-e_log-avx.c = -msse2 -mavx2 -DSSE2AVX
CFLAGS-mpa-avx.c = -msse2 -mavx2 -DSSE2AVX
CFLAGS-s_atan-avx.c = -msse2 -mavx2 -DSSE2AVX
CFLAGS-s_sin-avx.c = -msse2 -mavx2 -DSSE2AVX
CFLAGS-s_tan-avx.c = -msse2 -mavx2 -DSSE2AVX
endif

endif

ifeq ($(subdir),mathvec)
libmvec-sysdep_routines += svml_d_cos2_core_sse4 svml_d_cos4_core_avx2 \
			   svml_d_cos8_core_avx512 svml_d_sin2_core_sse4 \
			   svml_d_sin4_core_avx2 svml_d_sin8_core_avx512 \
			   svml_d_log2_core_sse4 svml_d_log4_core_avx2 \
			   svml_d_log8_core_avx512 svml_d_sincos2_core_sse4 \
			   svml_d_sincos4_core_avx2 svml_d_sincos8_core_avx512 \
			   svml_s_cosf4_core_sse4 svml_s_cosf8_core_avx2 \
			   svml_s_cosf16_core_avx512 svml_s_sinf4_core_sse4 \
			   svml_s_sinf8_core_avx2 svml_s_sinf16_core_avx512 \
			   svml_s_logf4_core_sse4 svml_s_logf8_core_avx2 \
			   svml_s_logf16_core_avx512 svml_d_exp2_core_sse4 \
			   svml_d_exp4_core_avx2 svml_d_exp8_core_avx512 \
			   svml_s_expf4_core_sse4 svml_s_expf8_core_avx2 \
			   svml_s_expf16_core_avx512 svml_d_pow2_core_sse4 \
			   svml_d_pow4_core_avx2 svml_d_pow8_core_avx512 \
			   svml_s_powf4_core_sse4 svml_s_powf8_core_avx2 \
			   svml_s_powf16_core_avx512 svml_s_sincosf4_core_sse4 \
			   svml_s_sincosf8_core_avx2 \
			   svml_s_sincosf16_core_avx512 \
			   svml_d_cos2_core-sse2 svml_d_cos4_core-sse \
			   svml_d_cos8_core-avx2 svml_d_exp2_core-sse2 \
			   svml_d_exp4_core-sse svml_d_exp8_core-avx2 \
			   svml_d_log2_core-sse2 svml_d_log4_core-sse \
			   svml_d_log8_core-avx2 svml_d_pow2_core-sse2 \
			   svml_d_pow4_core-sse svml_d_pow8_core-avx2 \
			   svml_d_sin2_core-sse2 svml_d_sin4_core-sse \
			   svml_d_sin8_core-avx2 \
			   svml_d_sincos2_core-sse2 \
			   svml_d_sincos4_core-sse \
			   svml_d_sincos8_core-avx2 \
			   svml_s_cosf16_core-avx2 \
			   svml_s_cosf4_core-sse2 \
			   svml_s_cosf8_core-sse \
			   svml_s_expf16_core-avx2 \
			   svml_s_expf4_core-sse2 \
			   svml_s_expf8_core-sse \
			   svml_s_logf16_core-avx2 \
			   svml_s_logf4_core-sse2 \
			   svml_s_logf8_core-sse \
			   svml_s_powf16_core-avx2 \
			   svml_s_powf4_core-sse2 \
			   svml_s_powf8_core-sse \
			   svml_s_sincosf16_core-avx2 \
			   svml_s_sincosf4_core-sse2 \
			   svml_s_sincosf8_core-sse \
			   svml_s_sinf16_core-avx2 \
			   svml_s_sinf4_core-sse2 \
			   svml_s_sinf8_core-sse
endif
