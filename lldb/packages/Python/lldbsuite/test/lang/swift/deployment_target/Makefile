LEVEL = ../../../make
SRCDIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

include $(LEVEL)/Makefile.rules

# This test only works on macOS 10.11+.

a.out: main.swift libNewerTarget.dylib
	$(SWIFTC) -target x86_64-apple-macosx10.10 -g -Onone $^ -lNewerTarget -L$(shell pwd) -o $@ $(SWIFTFLAGS) -I$(shell pwd) -Xfrontend -disable-target-os-checking

lib%.dylib: %.swift
	$(SWIFTC) -target x86_64-apple-macosx10.11 -g -Onone $^ -emit-library -module-name $(shell basename $< .swift) -emit-module -Xlinker -install_name -Xlinker @executable_path/$@

clean::
	rm -rf *.swiftmodule *.swiftdoc *.dSYM *~ lib*.dylib a.out *.o
