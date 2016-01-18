LEVEL := ../../make

LIB_PREFIX := loadunload_

DYLIB_EXECUTABLE_PATH := $(CURDIR)

DYLIB_NAME := $(LIB_PREFIX)d
DYLIB_C_SOURCES := d.c
DYLIB_ONLY := YES

include $(LEVEL)/Makefile.rules
