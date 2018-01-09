LEVEL = ../../../../../make

CXX_SOURCES := main.cpp

CFLAGS_EXTRAS += -O0
USE_LIBSTDCPP := 1

# clang-3.5+ outputs FullDebugInfo by default for Darwin/FreeBSD 
# targets.  Other targets do not, which causes this test to fail.
# This flag enables FullDebugInfo for all targets.
ifneq (,$(findstring clang,$(CC)))
  CFLAGS_EXTRAS += -fno-limit-debug-info
endif

include $(LEVEL)/Makefile.rules
