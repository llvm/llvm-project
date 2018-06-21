LEVEL := ../../../make

DYLIB_NAME := a
DYLIB_CXX_SOURCES := a.cpp
DYLIB_ONLY := YES

CXXFLAGS += -fPIC

include $(LEVEL)/Makefile.rules
