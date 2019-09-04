LEVEL = ../../make
DYLIB_ONLY := YES
DYLIB_NAME := $(BASENAME)
DYLIB_C_SOURCES := $(DYLIB_NAME).c

include $(LEVEL)/Makefile.rules
