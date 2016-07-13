LEVEL := ../../../make

DYLIB_NAME := b
DYLIB_CXX_SOURCES := b.cpp
DYLIB_ONLY := YES

CXXFLAGS += -fPIC

include $(LEVEL)/Makefile.rules
