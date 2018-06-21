LEVEL := ../../make

LIB_PREFIX := loadunload_

DYLIB_NAME := $(LIB_PREFIX)c
DYLIB_CXX_SOURCES := c.cpp
DYLIB_ONLY := YES

CXXFLAGS += -fPIC

include $(LEVEL)/Makefile.rules
