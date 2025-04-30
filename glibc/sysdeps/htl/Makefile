ifeq ($(subdir),htl)
libpthread-sysdep_routines += errno-loc
endif

ifeq ($(subdir),rt)
librt-sysdep_routines += timer_routines
endif

ifeq ($(subdir),posix)
CFLAGS-confstr.c += -DLIBPTHREAD_VERSION='"HTL $(version)"'
endif
