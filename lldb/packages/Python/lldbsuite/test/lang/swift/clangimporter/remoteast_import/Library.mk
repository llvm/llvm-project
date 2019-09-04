LEVEL = ../../../../make
DYLIB_SWIFT_SOURCES := Library.swift
DYLIB_NAME := Library
DYLIB_ONLY := YES

include $(LEVEL)/Makefile.rules

SWIFTFLAGS += -I.
