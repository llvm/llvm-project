LEVEL = ../../../make

all: libPlaygroundsRuntime.dylib PlaygroundStub

EXE=PlaygroundStub
SWIFT_SOURCES=PlaygroundStub.swift
LINKFLAGS=-lPlaygroundsRuntime -L$(shell pwd) -I$(shell pwd) 
SWIFTFLAGS=-g -Onone $(LINKFLAGS)

include $(LEVEL)/Makefile.rules

# Force a lower deployment target.
MACOSX_DEPLOYMENT_TARGET := 10.10
SWIFT_TRIPLE := -target x86_64-apple-macosx$(MACOSX_DEPLOYMENT_TARGET)

SWIFTFLAGS += $(SWIFT_TRIPLE)
SWIFTFLAGS_NOLINK=$(subst $(LINKFLAGS),,$(SWIFTFLAGS))

libPlaygroundsRuntime.dylib: PlaygroundsRuntime.swift
	$(SWIFTC) $(SWIFTFLAGS_NOLINK) -module-name PlaygroundsRuntime -emit-library -o $@ -Xlinker -install_name -Xlinker @executable_path/$@ $< 
ifneq "$(CODESIGN)" ""
	$(CODESIGN) -s - "$@"
endif

clean::
	rm -rf *.dSYM *.dylib *.swiftdoc *.swiftmodule PlaygroundStub libPlaygroundsRuntime.dylib

