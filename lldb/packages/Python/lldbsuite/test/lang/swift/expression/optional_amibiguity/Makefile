LEVEL = ../../../../make
SRCDIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
SWIFT_OBJC_INTEROP := 1

SWIFT_SOURCES := main.swift

include $(LEVEL)/Makefile.rules

a.out: main.swift
	$(SWIFTC) $(SWIFTFLAGS) -sdk "$(SWIFTSDKROOT)" $^ -emit-executable \
            -o a.out \
	    -emit-module -Xcc -I$(SRCDIR)

clean::
	rm -rf *.swiftmodule *.swiftdoc *.dSYM *~ lib*.dylib a.out *.o
