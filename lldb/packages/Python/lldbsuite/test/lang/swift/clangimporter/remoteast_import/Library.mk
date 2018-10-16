LEVEL = ../../../../make
SRCDIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
DYLIB_SWIFT_SOURCES := Library.swift
DYLIB_NAME := Library
DYLIB_ONLY := YES

include $(LEVEL)/Makefile.rules

SWIFTFLAGS += -I.
