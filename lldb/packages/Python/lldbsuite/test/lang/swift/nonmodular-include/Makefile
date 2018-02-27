LEVEL = ../../../make
SRCDIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))/
SWIFT_OBJC_INTEROP := 1
SWIFT_SOURCES := main.swift

include $(LEVEL)/Makefile.rules
SWIFTFLAGS += -Xcc -F$(SRCDIR) -I$(SRCDIR)

# Create a nonmodular include by removing the modulemap after building.
all: a.out
	rm module.modulemap

a.out: module.modulemap

module.modulemap:
	echo 'module Baz { header "conflict.h" }' >module.modulemap
