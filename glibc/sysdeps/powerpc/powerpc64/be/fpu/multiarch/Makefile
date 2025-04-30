ifeq ($(subdir),math)
# These functions are built both for libc and libm because they're required
# by printf.  While the libc objects have the prefix s_, the libm ones are
# prefixed with  m_.
sysdep_calls := s_modf-power5+ \
		s_modf-ppc64 \
		s_modff-power5+ \
		s_modff-ppc64

sysdep_routines += $(sysdep_calls)
libm-sysdep_routines += s_ceil-power5+ \
			s_ceil-ppc64 \
			s_ceilf-power5+ \
			s_ceilf-ppc64 \
			s_floor-power5+ \
			s_floor-ppc64 \
			s_floorf-power5+ \
			s_floorf-ppc64 \
			s_round-power5+ \
			s_round-ppc64 \
			s_roundf-power5+ \
			s_roundf-ppc64 \
			s_trunc-power5+ \
			s_trunc-ppc64 \
			s_truncf-power5+ \
			s_truncf-ppc64 \
			s_llrint-power8 \
			s_llrint-power6x \
			s_llrint-ppc64 \
			s_llround-power8 \
			s_llround-power6x \
			s_llround-power5+ \
			s_llround-ppc64 \
			s_llroundf-ppc64 \
			s_logb-power7 \
			s_logbf-power7 \
			s_logbl-power7 \
			s_logb-ppc64 \
			s_logbf-ppc64 \
			s_logbl-ppc64 \
			$(sysdep_calls:s_%=m_%)

CFLAGS-s_ceil-power5+.c = -mcpu=power5+
CFLAGS-s_ceilf-power5+.c = -mcpu=power5+
CFLAGS-s_floor-power5+.c = -mcpu=power5+
CFLAGS-s_floorf-power5+.c = -mcpu=power5+
CFLAGS-s_round-power5+.c = -mcpu=power5+
CFLAGS-s_roundf-power5+.c = -mcpu=power5+
CFLAGS-s_trunc-power5+.c = -mcpu=power5+
CFLAGS-s_truncf-power5+.c = -mcpu=power5+
CFLAGS-s_llrint-power8.c += -mcpu=power8
CFLAGS-s_llrint-power6x.c += -mcpu=power6x
CFLAGS-s_llround-power8.c += -mcpu=power8
CFLAGS-s_llround-power6x.c += -mcpu=power6x
CFLAGS-s_llround-power5+.c += -mcpu=power5+

CFLAGS-s_modf-power5+.c += -mcpu=power5+
CFLAGS-s_modff-power5+.c += -mcpu=power5+
CFLAGS-s_logbf-power7.c = -mcpu=power7
CFLAGS-s_logbl-power7.c = -mcpu=power7
CFLAGS-s_logb-power7.c = -mcpu=power7

# These files quiet sNaNs in a way that is optimized away without
# -fsignaling-nans.
CFLAGS-s_modf-ppc64.c += -fsignaling-nans
CFLAGS-s_modff-ppc64.c += -fsignaling-nans
endif
