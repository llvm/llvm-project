LEVEL := ../../make

DYLIB_NAME := dllfunc
DYLIB_C_SOURCES := dllfunc.c
DYLIB_ONLY := YES

include $(LEVEL)/Makefile.rules
