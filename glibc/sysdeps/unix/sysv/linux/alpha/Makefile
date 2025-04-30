ifeq ($(subdir),stdlib)
gen-as-const-headers += ucontext-offsets.sym
endif

ifeq ($(subdir),misc)
sysdep_headers += alpha/ptrace.h alpha/regdef.h sys/io.h

sysdep_routines += ieee_get_fp_control ieee_set_fp_control \
		   ioperm

# Support old timeval32 entry points
sysdep_routines += osf_adjtime osf_gettimeofday osf_settimeofday \
		   osf_getitimer osf_setitimer osf_utimes \
		   osf_getrusage osf_wait4

CFLAGS-ioperm.c = -Wa,-mev6
endif

ifeq ($(subdir),signal)
sysdep_routines += rt_sigaction
endif

ifeq ($(subdir),math)
# These 2 routines are normally in libgcc{.a,_s.so.1}.
# However, alpha -mlong-double-128 libgcc relies on
# glibc providing _Ots* routines and without these files
# glibc relies on __multc3/__divtc3 only provided
# by libgcc if configured with -mlong-double-128.
# Provide these routines here as well.
libm-routines += multc3 divtc3
endif   # math

ifeq ($(subdir),conform)
# For bug 21260.
conformtest-xfail-conds += alpha-linux
endif
