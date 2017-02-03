LEVEL = ../../../make

include $(LEVEL)/Makefile.rules

MACOSX_DEPLOYMENT_TARGET ?= 10.10
SWIFT_TRIPLE ?= -target x86_64-apple-macosx$(MACOSX_DEPLOYMENT_TARGET)
SDK_PATH ?= $(shell xcrun --show-sdk-path --sdk macosx)

everything: moda modb main

moda:
	$(SWIFTCC) $(SWIFT_TRIPLE) -sdk $(SDK_PATH) -g -Onone -Xfrontend -serialize-debugging-options -emit-module -module-name moda moda.swift -emit-library -o libmoda.dylib

modb:
	$(SWIFTCC) $(SWIFT_TRIPLE) -sdk $(SDK_PATH) -g -Onone -Xfrontend -serialize-debugging-options -emit-module -module-name modb modb.swift -emit-library -o libmodb.dylib  -lmoda -L. -I.

main:
	$(SWIFTCC) $(SWIFT_TRIPLE) -sdk $(SDK_PATH)  -g -Onone main.swift -o a.out -L. -I. -lmoda -lmodb

cleanup:
	rm -rf a.out.dSYM a.out libmoda.dylib libmoda.dylib.dSYM libmodb.dylib libmodb.dylib.dSYM moda.swiftdoc moda.swiftmodule modb.swiftmodule modb.swiftdoc

