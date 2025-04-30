ifeq ($(subdir),math)
sysdep_routines += s_modf-power5+ s_modf-ppc32 \
		   s_modff-power5+ s_modff-ppc32

libm-sysdep_routines += s_llrintf-power6 s_llrintf-ppc32 s_llrint-power6 \
			s_llrint-ppc32 s_llround-power6 s_llround-power5+ \
			s_llround-ppc32 s_ceil-power5+ \
			s_ceil-ppc32 s_ceilf-power5+ s_ceilf-ppc32 \
			s_floor-power5+ s_floor-ppc32 s_floorf-power5+ \
			s_floorf-ppc32 s_round-power5+ s_round-ppc32 \
			s_roundf-power5+ s_roundf-ppc32 s_trunc-power5+ \
			s_trunc-ppc32 s_truncf-power5+ s_truncf-ppc32 \
			s_lround-power6x \
			s_lround-power5+ s_lround-ppc32 s_lrint-power6x \
			s_lrint-ppc32 s_modf-power5+ s_modf-ppc32 \
			s_modff-power5+ s_modff-ppc32 s_logbl-power7 \
			s_logbl-ppc32 s_logb-power7 s_logb-ppc32 \
			s_logbf-power7 s_logbf-ppc32 e_hypot-power7 \
			e_hypot-ppc32 e_hypotf-power7 e_hypotf-ppc32

CFLAGS-s_llrintf-power6.c += -mcpu=power6
CFLAGS-s_llrintf-ppc32.c += -mcpu=power4
CFLAGS-s_llrint-power6.c += -mcpu=power6
CFLAGS-s_llrint-ppc32.c += -mcpu=power4
CFLAGS-s_lrint-ppc32.c += -mcpu=power4
CFLAGS-s_llround-power6.c += -mcpu=power6
CFLAGS-s_llround-power5+.c += -mcpu=power5+
CFLAGS-s_llround-ppc32.c += -mcpu=power4
CFLAGS-s_lround-ppc32.c += -mcpu=power4
CFLAGS-s_lround-power5+.c += -mcpu=power5+
CFLAGS-s_ceil-power5+.c = -mcpu=power5+
CFLAGS-s_ceilf-power5+.c = -mcpu=power5+
CFLAGS-s_modf-power5+.c = -mcpu=power5+
CFLAGS-s_modff-power5+.c = -mcpu=power5+
CFLAGS-s_logbl-power7.c = -mcpu=power7
CFLAGS-s_logb-power7.c = -mcpu=power7
CFLAGS-s_logbf-power7.c = -mcpu=power7
CFLAGS-e_hypot-power7.c = -mcpu=power7
CFLAGS-e_hypotf-power7.c = -mcpu=power7

# These files quiet sNaNs in a way that is optimized away without
# -fsignaling-nans.
CFLAGS-s_modf-ppc32.c += -fsignaling-nans
CFLAGS-s_modff-ppc32.c += -fsignaling-nans
endif
