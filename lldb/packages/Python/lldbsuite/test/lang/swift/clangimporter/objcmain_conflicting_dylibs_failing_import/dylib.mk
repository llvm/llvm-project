LEVEL = ../../../../make
DYLIB_ONLY := YES
DYLIB_NAME := $(BASENAME)
DYLIB_SWIFT_SOURCES := $(DYLIB_NAME).swift
SWIFTFLAGS_EXTRAS = \
  -Xcc -I$(SRCDIR) -Xcc -I$(SRCDIR)/$(BASENAME) \
  -emit-objc-header-path $(BASENAME).h

include $(LEVEL)/Makefile.rules
