LEVEL = ../../../make
DYLIB_ONLY := YES
DYLIB_SWIFT_SOURCES := $(DYLIB_NAME).swift
SWIFTFLAGS_EXTRAS = -I$(shell pwd) -enable-library-evolution

include $(LEVEL)/Makefile.rules
