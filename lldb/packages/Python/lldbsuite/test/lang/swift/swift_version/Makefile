LEVEL = ../../../make

include $(LEVEL)/Makefile.rules

MACOSX_DEPLOYMENT_TARGET ?= 10.10
SWIFT_TRIPLE ?= -target x86_64-apple-macosx$(MACOSX_DEPLOYMENT_TARGET)
SDK_PATH ?= $(shell xcrun --show-sdk-path --sdk macosx)

everything: libmod3.dylib libmod4.dylib main

MOD3_FLAGS = -swift-version 3
MOD4_FLAGS = -swift-version 4

libmod3.dylib: mod3.swift
	$(SWIFTCC) $(SWIFT_TRIPLE) -sdk $(SDK_PATH) -g -Onone -Xfrontend -serialize-debugging-options $(MOD3_FLAGS) -emit-module -module-name mod3 mod3.swift -emit-library -o libmod3.dylib

libmod4.dylib: mod4.swift
	$(SWIFTCC) $(SWIFT_TRIPLE) -sdk $(SDK_PATH) -g -Onone -Xfrontend -serialize-debugging-options $(MOD4_FLAGS) -emit-module -module-name mod4 mod4.swift -emit-library -o libmod4.dylib

main:
	$(SWIFTCC) $(SWIFT_TRIPLE) -sdk $(SDK_PATH)  -g -Onone $(RESILIENCE_FLAGS) main.swift -o main -L. -I. -lmod3 -lmod4

cleanup:
	rm -rf main main.a main.b *.dylib *.dSYM *.swiftdoc *.swiftmodule mod.swift

