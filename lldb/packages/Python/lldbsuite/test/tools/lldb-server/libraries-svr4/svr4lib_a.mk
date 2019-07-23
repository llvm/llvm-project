LEVEL = ../../../make

LIB_PREFIX := svr4lib

DYLIB_NAME := $(LIB_PREFIX)_a
DYLIB_CXX_SOURCES := $(LIB_PREFIX)_a.cpp
DYLIB_ONLY := YES

include $(LEVEL)/Makefile.rules
