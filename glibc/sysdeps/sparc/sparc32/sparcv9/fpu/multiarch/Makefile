ifeq ($(subdir),math)

# These functions are built both for libc and libm because they're required
# by printf.  While the libc objects have the prefix s_, the libm ones are
# prefixed with  m_.
sysdep_calls := s_copysignf-vis3 s_copysign-vis3 s_copysignf-generic \
		s_copysign-generic

sysdep_routines += $(sysdep_calls)
libm-sysdep_routines += s_fabs-vis3 s_fabsf-vis3 s_fabs-generic \
			s_fabsf-generic s_llrintf-vis3 s_llrint-vis3 \
			s_llrintf-generic s_llrint-generic \
			s_rintf-vis3 s_rint-vis3 s_rintf-generic \
			s_rint-generic s_fmaf-vis3 s_fma-vis3 s_fma-generic \
			s_fmaf-generic s_nearbyint-vis3 s_nearbyintf-vis3 \
			s_nearbyint-generic s_nearbyintf-generic \
			s_fdimf-vis3 s_fdim-vis3 s_fdim-generic \
			s_fdimf-generic \
			$(sysdep_calls:s_%=m_%)

CFLAGS-s_fdimf-vis3.c += -Wa,-Av9d -mvis3
CFLAGS-s_fdim-vis3.c += -Wa,-Av9d -mvis3
endif
