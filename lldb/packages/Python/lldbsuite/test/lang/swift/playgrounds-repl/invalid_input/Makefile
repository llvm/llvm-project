LEVEL = ../../../../make

EXE = PlaygroundStub
SWIFT_SOURCES = PlaygroundStub.swift

include $(LEVEL)/Makefile.rules

MAIN_SWIFTFLAGS = $(SWIFTFLAGS) -lPlaygroundsRuntime -L. -I.

all: libPlaygroundsRuntime.dylib PlaygroundStub

libPlaygroundsRuntime.dylib: PlaygroundsRuntime.swift
	$(SWIFTC) $(SWIFTFLAGS) -Onone -module-name PlaygroundsRuntime -emit-library -o $@ $<

PlaygroundStub:  libPlaygroundsRuntime.dylib

clean::
	rm -rf *.dSYM *.dylib *.swiftdoc *.swiftmodule PlaygroundStub
