ifeq ($(subdir),math)
libm-sysdep_routines += e_exp2f-sse2 e_expf-sse2 e_logf-sse2 e_log2f-sse2 \
			e_powf-sse2 s_sinf-sse2 s_cosf-sse2 s_sincosf-sse2

CFLAGS-e_exp2f-sse2.c = -msse2 -mfpmath=sse
CFLAGS-e_expf-sse2.c = -msse2 -mfpmath=sse
CFLAGS-e_log2f-sse2.c = -msse2 -mfpmath=sse
CFLAGS-e_logf-sse2.c = -msse2 -mfpmath=sse
CFLAGS-e_powf-sse2.c = -msse2 -mfpmath=sse
endif
