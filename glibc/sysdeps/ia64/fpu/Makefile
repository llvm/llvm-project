ifeq ($(subdir),math)
#
# Some files which need to go both into libc and libm have external
# dependencies which need to be resolved differently for libc
# vs. libm.  For example, inside libc, __libm_error_support needs to
# resolve to HIDDEN_JUMPTARGET(__libm_error_support) whereas within
# libm it always resolves to __libm_error_support.  Such files need to
# be compiled twice.  Fortunately, math/Makefile already has logic to
# support this: if a file starts with "s_", make will automatically
# generate a matching file whose name starts with "m_" which simply
# includes the corresponding "s_" file.
#
duplicated-routines = s_libm_ldexp s_libm_ldexpf s_libm_ldexpl \
		      s_libm_scalbn s_libm_scalbnf s_libm_scalbnl

libm-sysdep_routines += s_erfc s_erfcf s_erfcl \
			s_matherrf s_matherrl libm_reduce \
			libm_error \
			libm_frexp libm_frexpf libm_frexpl \
			libm_sincos libm_sincosf libm_sincosl \
			libm_sincos_large \
			libm_lgamma libm_lgammaf libm_lgammal \
			libm_scalblnf \
			$(duplicated-routines:s_%=m_%)

sysdep_routines += libc_libm_error libm_frexp libm_frexpf libm_frexpl \
		   $(duplicated-routines)

sysdep-CPPFLAGS += -include libm-symbols.h \
	-D__POSIX__ -Dopensource \
	-D_LIB_VERSIONIMF=_LIB_VERSION \
	-DSIZE_INT_32 -DSIZE_LONG_INT_64 -DSIZE_LONG_LONG_INT_64 \
	-DSIZE_LONG_64 -DIA64
endif
