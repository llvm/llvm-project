LEVEL = ../../../make

include $(LEVEL)/Makefile.rules

MACOSX_DEPLOYMENT_TARGET ?= 10.10
SWIFT_TRIPLE ?= -target x86_64-apple-macosx$(MACOSX_DEPLOYMENT_TARGET)
SDK_PATH ?= $(shell xcrun --show-sdk-path --sdk macosx)

everything: UnitTest.xctest/Contents/MacOS/test UnitTest.xctest/Contents/Info.plist xctest

UnitTest.xctest/Contents/MacOS/test: test.swift
	$(SWIFTCC) $(SWIFT_TRIPLE) -sdk $(SDK_PATH) -g -Onone -module-name test -emit-library -o test test.swift
	mkdir -p UnitTest.xctest/Contents/MacOS
	mv test UnitTest.xctest/Contents/MacOS/test
	mv test.dSYM UnitTest.xctest/Contents/MacOS/test.dSYM

UnitTest.xctest/Contents/Info.plist: Info.plist
	cp Info.plist UnitTest.xctest/Contents/

xctest:
	$(CC) $(CFLAGS) -o xctest xctest.c

cleanup:
	rm -rf *.dSYM *.dylib *.swiftdoc *.swiftmodule *.xctest xctest

