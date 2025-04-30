ifeq ($(subdir),math)
# These functions are built both for libc and libm because they're required
# by printf.  While the libc objects have the prefix s_, the libm ones are
# prefixed with  m_.
sysdep_calls := s_signbitf-vis3 s_signbit-vis3 s_signbitf-generic \
		s_signbit-generic s_finitef-vis3 s_finite-vis3 \
		s_finitef-generic s_finite-generic \
		s_isinff-vis3 s_isinf-vis3 s_isinff-generic s_isinf-generic \
		s_isnanf-vis3 s_isnan-vis3 s_isnanf-generic s_isnan-generic

sysdep_routines += $(sysdep_calls)
libm-sysdep_routines += s_lrintf-vis3 s_lrint-vis3 s_lrintf-generic \
			s_lrint-generic s_rintf-vis3 s_rint-vis3 \
			s_rintf-generic s_rint-generic \
			s_fmaf-vis3 s_fma-vis3 s_fmaf-generic s_fma-generic \
			s_nearbyint-vis3 s_nearbyintf-vis3 \
			s_nearbyint-generic s_nearbyintf-generic \
			s_ceilf-vis3 s_ceil-vis3 s_ceilf-generic \
			s_ceil-generic s_floorf-vis3 s_floor-vis3 \
			s_floorf-generic s_floor-generic s_truncf-vis3 \
			s_trunc-vis3 s_truncf-generic s_trunc-generic \
			$(sysdep_calls:s_%=m_%)

CFLAGS-s_ceilf-vis3.c += -Wa,-Av9d -mvis3
CFLAGS-s_ceil-vis3.c += -Wa,-Av9d -mvis3
CFLAGS-s_floorf-vis3.c += -Wa,-Av9d -mvis3
CFLAGS-s_floor-vis3.c += -Wa,-Av9d -mvis3
CFLAGS-s_truncf-vis3.c += -Wa,-Av9d -mvis3
CFLAGS-s_trunc-vis3.c += -Wa,-Av9d -mvis3
endif
