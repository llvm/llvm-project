CXX_SOURCES := main.cpp

ifneq (,$(findstring icc,$(CC)))
    CXXFLAGS += -debug inline-debug-info
endif

include Makefile.rules
