LEVEL = ../../../make

all: UnitTest.xctest/Contents/MacOS/test UnitTest.xctest/Contents/Info.plist xctest

C_SOURCES=xctest.c
EXE=xctest
CFLAGS=-O0 -g -Wl,-rpath,$(shell pwd)
include $(LEVEL)/Makefile.rules

MACOSX_DEPLOYMENT_TARGET ?= 10.10
SWIFT_TRIPLE ?= -target x86_64-apple-macosx$(MACOSX_DEPLOYMENT_TARGET)
SDK_PATH ?= $(shell xcrun --show-sdk-path --sdk macosx)

UnitTest.xctest/Contents/MacOS/test: test.swift
	$(SWIFTCC) $(SWIFTFLAGS) $(SWIFT_TRIPLE) -sdk $(SDK_PATH) -module-name test -emit-library -o test $<
	mkdir -p UnitTest.xctest/Contents/MacOS
	mv test UnitTest.xctest/Contents/MacOS/test
	mv test.dSYM UnitTest.xctest/Contents/MacOS/test.dSYM

UnitTest.xctest/Contents/Info.plist: Info.plist
	cp $< UnitTest.xctest/Contents/

clean::
	rm -rf *.dSYM *.dylib *.swiftdoc *.swiftmodule *.xctest xctest

