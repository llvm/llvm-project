C_SOURCES := main.c

ifneq (,$(findstring icc,$(CC)))
    CFLAGS += -debug inline-debug-info
endif

include Makefile.rules
