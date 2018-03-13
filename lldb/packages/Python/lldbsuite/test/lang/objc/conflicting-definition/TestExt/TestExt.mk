LEVEL = ../../../make

DYLIB_NAME := TestExt
DYLIB_ONLY := YES
CFLAGS_EXTRAS = -I$(SRCDIR)/..
LD_EXTRAS = -L. -lTest -lobjc -framework Foundation

DYLIB_OBJC_SOURCES = TestExt/TestExt.m

include $(LEVEL)/Makefile.rules
