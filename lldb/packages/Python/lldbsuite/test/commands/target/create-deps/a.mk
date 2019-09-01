LEVEL := ../../make

LIB_PREFIX := load_

DYLIB_NAME := $(LIB_PREFIX)a
DYLIB_CXX_SOURCES := a.cpp
DYLIB_ONLY := YES

include $(LEVEL)/Makefile.rules
