LEVEL = ../../../make
DYLIB_ONLY := YES
DYLIB_NAME := $(BASENAME)
DYLIB_SWIFT_SOURCES := $(DYLIB_NAME).swift
SWIFTFLAGS_EXTRAS = -I$(shell pwd)

include $(LEVEL)/Makefile.rules
