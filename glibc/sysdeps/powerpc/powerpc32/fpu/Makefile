ifeq ($(subdir),math)
# lrint is aliased to lrintf, so suppress compiler builtins to
# avoid mismatched signatures.
CFLAGS-s_lrint.c += -fno-builtin-lrintf
CFLAGS-s_lround.c += -fno-builtin-lroundf
endif

ifeq ($(subdir),misc)
sysdep_routines += fprsave fprrest
endif
