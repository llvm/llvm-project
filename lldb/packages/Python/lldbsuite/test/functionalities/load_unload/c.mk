LEVEL := ../../make

LIB_PREFIX := loadunload_

DYLIB_NAME := $(LIB_PREFIX)c
DYLIB_C_SOURCES := c.c
DYLIB_ONLY := YES

include $(LEVEL)/Makefile.rules
