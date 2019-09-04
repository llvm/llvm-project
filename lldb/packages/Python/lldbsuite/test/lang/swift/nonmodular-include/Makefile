LEVEL = ../../../make
SWIFT_OBJC_INTEROP := 1
SWIFT_SOURCES := main.swift

all: module.modulemap conflict.h a.out

include $(LEVEL)/Makefile.rules
SWIFTFLAGS += -Xcc -F$(SRCDIR) -I$(SRCDIR) -I$(BUILDDIR)

# Create a nonmodular include by removing the modulemap after building.
all:
	rm module.modulemap

conflict.h: conflict.h.in
	cp $< $@

module.modulemap:
	echo 'module Baz { header "conflict.h" }' >$@
