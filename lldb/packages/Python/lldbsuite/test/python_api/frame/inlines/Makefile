C_SOURCES := inlines.c

ifneq (,$(findstring icc,$(CC)))
    CFLAGS += -debug inline-debug-info
endif

include Makefile.rules
