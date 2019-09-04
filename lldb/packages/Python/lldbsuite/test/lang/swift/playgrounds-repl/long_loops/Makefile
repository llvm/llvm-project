LEVEL = ../../../../make

EXE = PlaygroundStub
SWIFT_SOURCES = PlaygroundStub.swift

include $(LEVEL)/Makefile.rules

MACOSX_DEPLOYMENT_TARGET ?= 10.10
SWIFT_TRIPLE ?= -target x86_64-apple-macosx$(MACOSX_DEPLOYMENT_TARGET)
SDK_PATH ?= $(shell xcrun --show-sdk-path --sdk macosx)

all: libPlaygroundsRuntime.dylib PlaygroundStub

libPlaygroundsRuntime.dylib: PlaygroundsRuntime.swift
	$(SWIFTC) $(SWIFT_TRIPLE) $(SWIFTFLAGS) -sdk $(SDK_PATH) -Onone -module-name PlaygroundsRuntime -emit-library -o $@ $<

PlaygroundStub:  libPlaygroundsRuntime.dylib

clean::
	rm -rf *.dSYM *.dylib *.swiftdoc *.swiftmodule PlaygroundStub

