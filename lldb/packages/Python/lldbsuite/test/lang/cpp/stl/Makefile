LEVEL = ../../../make

CXX_SOURCES := main.cpp
CFLAGS := -g -O0

clean: OBJECTS+=$(wildcard main.d.*)

# clang-3.5+ outputs FullDebugInfo by default for Darwin/FreeBSD 
# targets.  Other targets do not, which causes this test to fail.
# This flag enables FullDebugInfo for all targets.
ifneq (,$(findstring clang,$(CC)))
  CFLAGS_EXTRAS += -fno-limit-debug-info
endif

include $(LEVEL)/Makefile.rules
