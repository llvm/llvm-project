LEVEL = ../../../make
SRCDIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))/
SWIFT_OBJC_INTEROP := 1
SWIFT_SOURCES := main.swift

all: module.modulemap conflict.h a.out

include $(LEVEL)/Makefile.rules
# $(PWD) is not what we want because VPATH changes it
SWIFTFLAGS += -Xcc -F$(SRCDIR) -I$(SRCDIR) -I$(shell pwd)

# Create a nonmodular include by removing the modulemap after building.
all:
	rm module.modulemap

conflict.h: conflict.h.in
	cp $< $@

module.modulemap:
	echo 'module Baz { header "conflict.h" }' >$@
