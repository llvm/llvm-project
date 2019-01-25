LEVEL = ../../../make

all: libPlaygroundsRuntime.dylib PlaygroundStub

EXE=PlaygroundStub
SWIFT_SOURCES=PlaygroundStub.swift
#LINKFLAGS=-lPlaygroundsRuntime -L$(shell pwd) -I$(shell pwd) 
#SWIFTFLAGS=-g -Onone $(LINKFLAGS)
# Force a lower deployment target.
MACOSX_DEPLOYMENT_TARGET := 10.10
TRIPLE=x86_64-apple-macosx$(MACOSX_DEPLOYMENT_TARGET)

include $(LEVEL)/Makefile.rules

LD_EXTRAS=-lPlaygroundsRuntime -L$(shell pwd)

libPlaygroundsRuntime.dylib: PlaygroundsRuntime.swift
	$(SWIFTC) $(SWIFTFLAGS_NOLINK) -module-name PlaygroundsRuntime -emit-library -o $@ -Xlinker -install_name -Xlinker @executable_path/$@ $<
ifneq "$(CODESIGN)" ""#
	$(CODESIGN) -s - "$@"
endif

clean::
	rm -rf *.dSYM *.dylib *.swiftdoc *.swiftmodule PlaygroundStub libPlaygroundsRuntime.dylib PlaygroundsRuntime.*

