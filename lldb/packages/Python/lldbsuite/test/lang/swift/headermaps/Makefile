LEVEL = ../../../make
SRCDIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))/
SWIFT_OBJC_INTEROP := 1
SWIFT_SOURCES := main.swift
include $(LEVEL)/Makefile.rules
SWIFTFLAGS += -Xcc -I$(SRCDIR)/foo.hmap -Xcc -I$(SRCDIR)
