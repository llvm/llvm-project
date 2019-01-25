LEVEL = ../../../make

all: a.out dlopen_module

include $(LEVEL)/Makefile.rules

# This test only works on macOS 10.11+.

a.out: main.swift libNewerTarget.dylib
	$(SWIFTC) -target x86_64-apple-macosx10.10 -g -Onone $^ -lNewerTarget -L$(shell pwd) -o $@ $(SWIFTFLAGS) -I$(shell pwd) -Xfrontend -disable-target-os-checking
ifneq "$(CODESIGN)" ""
	$(CODESIGN) -s - "$@"
endif

dlopen_module: main.m libNewerTarget.dylib
	$(CC) -framework Foundation $^ -mmacos-version-min=10.8 -o $@ $(CFLAGS)
ifneq "$(CODESIGN)" ""
	$(CODESIGN) -s - "$@"
endif

lib%.dylib: %.swift
	$(SWIFTC) -target x86_64-apple-macosx10.11 -g -Onone $^ -emit-library -module-name $(shell basename $< .swift) -emit-module -Xlinker -install_name -Xlinker @executable_path/$@
ifneq "$(CODESIGN)" ""
	$(CODESIGN) -s - "$@"
endif

clean::
	rm -rf *.swiftmodule *.swiftdoc *.dSYM *~ lib*.dylib a.out dlopen_module *.o
