LEVEL = ../../../make
# We don't have any standard include directories, so we can't
# parse the test_common.h header we usually inject as it includes
# system headers.
NO_TEST_COMMON_H := 1

CXXFLAGS += $(MANDATORY_CXXMODULE_BUILD_CFLAGS)
CXXFLAGS += -I $(SRCDIR)/root/usr/include/c++/include/ -I $(SRCDIR)/root/usr/include/ -nostdinc -nostdinc++ -nostdlib++
CXX_SOURCES := main.cpp
include $(LEVEL)/Makefile.rules
