C_SOURCES := main.c

ifeq "$(OS)" ""
    OS = $(shell uname -s)
endif

ifeq "$(OS)" "Darwin"
    LD_EXTRAS = -Xlinker -dead_strip
else ifeq "$(OS)" "Windows_NT"
    CFLAGS_EXTRAS += /Gw /Gy
    LD_EXTRAS = -Xlinker /OPT:REF
else
    CFLAGS_EXTRAS += -fdata-sections -ffunction-sections
    LD_EXTRAS = -Wl,--gc-sections
endif

MAKE_DSYM := NO

include Makefile.rules
