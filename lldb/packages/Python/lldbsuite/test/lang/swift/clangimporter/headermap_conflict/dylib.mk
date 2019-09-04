LEVEL = ../../../../make
DYLIB_ONLY := YES
DYLIB_NAME := $(BASENAME)
DYLIB_SWIFT_SOURCES := $(DYLIB_NAME).swift
SWIFTFLAGS_EXTRAS = -Xcc -I$(SRCDIR)/foo.hmap  -Xcc -I$(SRCDIR)

include $(LEVEL)/Makefile.rules
