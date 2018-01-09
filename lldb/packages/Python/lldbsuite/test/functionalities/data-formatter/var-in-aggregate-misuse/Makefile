LEVEL = ../../../make

CXX_SOURCES := main.cpp

# Workaround for llvm.org/pr16214: clang doesn't emit structure definition DWARF
# information without the flag below.
ifneq (,$(findstring clang,$(CC)))
  CFLAGS_EXTRAS += -fno-limit-debug-info
endif

include $(LEVEL)/Makefile.rules
