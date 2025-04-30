ifeq ($(subdir),signal)
# sigaction.c defines static functions in asms and refers to them from
# C code, resulting in "'restore_rt' used but never defined" (which
# has no corresponding warning option to allow it to be disabled with
# diagnostic pragmas).
CFLAGS-sigaction.c += -Wno-error
endif

ifeq ($(subdir),nptl)
CFLAGS-sigaction.c += -Wno-error
endif
