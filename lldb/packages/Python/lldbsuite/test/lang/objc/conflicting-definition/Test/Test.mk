LEVEL = ../../../make

DYLIB_NAME := Test
DYLIB_ONLY := YES
CFLAGS_EXTRAS = -I$(SRCDIR)/..
LD_EXTRAS = -lobjc -framework Foundation

DYLIB_OBJC_SOURCES = Test/Test.m

include $(LEVEL)/Makefile.rules
