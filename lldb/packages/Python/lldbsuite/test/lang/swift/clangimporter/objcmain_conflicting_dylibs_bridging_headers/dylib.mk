LEVEL = ../../../../make
DYLIB_ONLY := YES
DYLIB_NAME := $(BASENAME)
DYLIB_SWIFT_SOURCES := $(DYLIB_NAME).swift
SWIFT_OBJC_INTEROP := 1

include $(LEVEL)/Makefile.rules

SWIFTFLAGS += -Xcc -I$(SRCDIR) \
  -emit-objc-header-path $(DYLIB_NAME)-Swift.h \
  -import-objc-header $(SRCDIR)/$(DYLIB_NAME)/$(DYLIB_NAME)-Bridging.h
