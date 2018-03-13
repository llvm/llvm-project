LEVEL = ../../../make

all: libPlaygroundsRuntime.dylib PlaygroundStub

EXE=PlaygroundStub
SWIFT_SOURCES=PlaygroundStub.swift
LINKFLAGS=-lPlaygroundsRuntime -L$(shell pwd) -I$(shell pwd) 
SWIFTFLAGS=-g -Onone $(LINKFLAGS)

include $(LEVEL)/Makefile.rules

SWIFTFLAGS_NOLINK=$(subst $(LINKFLAGS),,$(SWIFTFLAGS))

MACOSX_DEPLOYMENT_TARGET ?= 10.10
SWIFT_TRIPLE ?= -target x86_64-apple-macosx$(MACOSX_DEPLOYMENT_TARGET)

libPlaygroundsRuntime.dylib: PlaygroundsRuntime.swift
	$(SWIFTCC) $(SWIFT_TRIPLE) $(SWIFTFLAGS_NOLINK) -module-name PlaygroundsRuntime -emit-library -o $@ -Xlinker -install_name -Xlinker @executable_path/$@ $< 

clean::
	rm -rf *.dSYM *.dylib *.swiftdoc *.swiftmodule PlaygroundStub libPlaygroundsRuntime.dylib

