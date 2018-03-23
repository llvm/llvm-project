LEVEL = ../../../make

EXE=main
SWIFT_SOURCES=main.swift
LINKFLAGS= -lmoda -lmodb -L$(shell pwd) -I$(shell pwd)
SWIFTFLAGS= -g -Onone $(LINKFLAGS) -Xfrontend -serialize-debugging-options 

all: libmoda.dylib libmodb.dylib main

include $(LEVEL)/Makefile.rules

SWIFTFLAGS_NOLINK = $(subst $(LINKFLAGS),,$(SWIFTFLAGS))

libmoda.dylib: moda.swift
	$(SWIFTCC) $(SWIFTFLAGS_NOLINK) -emit-module -module-name moda $< -emit-library -o $@ -Xlinker -install_name -Xlinker @executable_path/$@ 

libmodb.dylib: modb.swift
	$(SWIFTCC) $(SWIFTFLAGS_NOLINK) -emit-module -module-name modb $< -emit-library -o $@ -lmoda -L$(shell pwd) -I$(shell pwd) -Xlinker -install_name -Xlinker @executable_path/$@

clean::
	rm -rf a.out.dSYM a.out libmoda.dylib libmoda.dylib.dSYM libmodb.dylib libmodb.dylib.dSYM moda.swiftdoc moda.swiftmodule modb.swiftmodule modb.swiftdoc

