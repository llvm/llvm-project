ifeq ($(subdir),stdlib)
gen-as-const-headers += ucontext_i.sym
endif

ifeq ($(subdir),misc)
# MIPS-style cacheflush routine
sysdep_headers += sys/cachectl.h
sysdep_routines += cacheflush
endif

ifeq ($(subdir),csu)
# Work around an incorrect assumption that an object of variably
# modified type, with a small fixed size but a large variable-sized
# part (the static_slotinfo variable in libc-tls.c) is allocated in
# the small data section, not the data section where it belongs based
# on its size.  See GCC PR 92499.
CFLAGS-libc-tls.c += -mgpopt=none
endif
