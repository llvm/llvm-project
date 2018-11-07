LEVEL = ../../../make

EXE=main
SWIFT_SOURCES=main.swift
LINKFLAGS=-L$(shell pwd) -I$(shell pwd) -lmod3 -lmod4
SWIFTFLAGS=-g -Onone $(LINKFLAGS) -Xfrontend -serialize-debugging-options

all: libmod3.dylib libmod4.dylib main

include $(LEVEL)/Makefile.rules

SWIFTFLAGS_NOLINK=$(subst $(LINKFLAGS),,$(SWIFTFLAGS))

MACOSX_DEPLOYMENT_TARGET ?= 10.10
SWIFT_TRIPLE ?= -target x86_64-apple-macosx$(MACOSX_DEPLOYMENT_TARGET)
SDK_PATH ?= $(shell xcrun --show-sdk-path --sdk macosx)

MOD3_FLAGS = -swift-version 3
MOD4_FLAGS = -swift-version 4

libmod%.dylib: mod%.swift
	$(SWIFTC) $(SWIFTFLAGS_NOLINK) $(SWIFT_TRIPLE) $(MOD$*_FLAGS) -emit-module -module-name mod$* $< -emit-library -o libmod$*.dylib -Xlinker -install_name -Xlinker @executable_path/libmod$*.dylib
ifneq "$(CODESIGN)" ""
	$(CODESIGN) -s - "$@"
endif


clean::
	rm -rf main main.a main.b *.dylib *.dSYM *.swiftdoc *.swiftmodule mod.swift

