LEVEL = ../../../../make
SRCDIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
SWIFT_SOURCES := main.swift
# This Makefile overwrites the default rules.
EXE := a.out

all: libLibrary a.out

include $(LEVEL)/Makefile.rules

SWIFTFLAGS += -I. -Xcc -DSYNTAX_ERROR=1
LD_EXTRAS += -lLibrary -L.

.PHONY: libLibrary

libLibrary:
	$(MAKE) -f $(SRCDIR)/Library.mk VPATH=$(SRCDIR) -I $(SRCDIR)

clean::
	rm -rf *.swiftmodule *.swiftdoc *.dSYM *~ lib*.dylib a.out *.o
