# Makefile fragment for POWER4/5/5+.

ifeq ($(subdir),string)
CFLAGS-wordcopy.c += --param max-variable-expansions-in-unroller=2 --param max-unroll-times=2 -funroll-loops -fpeel-loops
CFLAGS-memmove.c += --param max-variable-expansions-in-unroller=2 --param max-unroll-times=2 -funroll-loops -fpeel-loops
endif
