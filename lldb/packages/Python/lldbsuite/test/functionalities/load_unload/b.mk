LEVEL := ../../make

LIB_PREFIX := loadunload_

DYLIB_NAME := $(LIB_PREFIX)b
DYLIB_C_SOURCES := b.c
DYLIB_ONLY := YES

include $(LEVEL)/Makefile.rules
