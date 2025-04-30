abi-variants := 32 64
abi-32-condition := __WORDSIZE == 32
abi-64-condition := __WORDSIZE == 64

ifeq ($(subdir),rt)
librt-routines += rt-sysdep
librt-shared-only-routines += rt-sysdep
endif

ifeq ($(subdir),stdlib)
gen-as-const-headers += ucontext_i.sym
endif

ifeq ($(subdir),nptl)
elision-CFLAGS = -mhtm -msoft-float
CFLAGS-elision-lock.c = $(elision-CFLAGS)
CFLAGS-elision-timed.c = $(elision-CFLAGS)
CFLAGS-elision-trylock.c = $(elision-CFLAGS)
CFLAGS-elision-unlock.c = $(elision-CFLAGS)
endif

ifeq ($(subdir),misc)
tests += tst-ptrace-singleblock
endif
