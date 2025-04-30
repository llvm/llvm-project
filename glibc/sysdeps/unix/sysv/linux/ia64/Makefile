ifeq ($(subdir),misc)
sysdep_headers += sys/rse.h
endif

ifeq ($(subdir),stdlib)
sysdep_routines += __start_context
gen-as-const-headers += sigcontext-offsets.sym
endif

ifeq ($(subdir),misc)
sysdep_headers += sys/io.h
sysdep_routines += ioperm clone2
gen-as-const-headers += sigaltstack-offsets.sym
endif

ifeq ($(subdir),rt)
librt-routines += rt-sysdep
librt-shared-only-routines += rt-sysdep
endif

ifeq ($(subdir),nptl)
sysdep_routines += __ia64_longjmp unwind_longjmp __sigstack_longjmp
endif

ifeq ($(subdir),conform)
# For bug 21634.
conformtest-xfail-conds += ia64-linux
endif
