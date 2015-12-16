LEVEL := ../../make

LIB_PREFIX := loadunload_

DYLIB_EXECUTABLE_PATH := $(CURDIR)

DYLIB_NAME := $(LIB_PREFIX)d
DYLIB_CXX_SOURCES := d.cpp
DYLIB_ONLY := YES

CXXFLAGS += -fPIC

include $(LEVEL)/Makefile.rules
