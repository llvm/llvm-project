LEVEL = ../../../make

include $(LEVEL)/Makefile.rules

MACOSX_DEPLOYMENT_TARGET ?= 10.10
SWIFT_TRIPLE ?= -target x86_64-apple-macosx$(MACOSX_DEPLOYMENT_TARGET)
SDK_PATH ?= $(shell xcrun --show-sdk-path --sdk macosx)

everything: libPlaygroundsRuntime.dylib PlaygroundStub

libPlaygroundsRuntime.dylib:
	$(SWIFTCC) $(SWIFT_TRIPLE) -sdk $(SDK_PATH) -Onone -module-name PlaygroundsRuntime -emit-library -o libPlaygroundsRuntime.dylib PlaygroundsRuntime.swift

PlaygroundStub: libPlaygroundsRuntime.dylib
	$(SWIFTCC) $(SWIFT_TRIPLE) -sdk $(SDK_PATH) -g -Onone PlaygroundStub.swift -o PlaygroundStub -lPlaygroundsRuntime -L. -I.

cleanup:
	rm -rf *.dSYM *.dylib *.swiftdoc *.swiftmodule PlaygroundStub

