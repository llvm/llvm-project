LEVEL = ../../../../make
SRCDIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
SWIFT_OBJC_INTEROP := 1
# This Makefile overwrites the default rules.
EXE=a.out

# This test builds a .dylib whose .swiftmodule imports a Clang module
# which uses a header file "Foo.h". The main swift program also
# imports a Clang module that imports a header file called "Foo.h",
# but it is in a different directory. We are using a headermap to
# switch between the two versions of "Foo.h" during build time.

include $(LEVEL)/Makefile.rules

.PHONY: libDylib

a.out: main.swift libDylib
#       Avoid building a dSYM for the main application.
#	$(SWIFTC) -g -Onone $^ -lDylib -L$(shell pwd) -o $@ -I$(SRCDIR) $(SWIFTFLAGS) -import-objc-header $(SRCDIR)/bridging-header.h -Xcc -I$(SRCDIR)/foo.hmap
	$(SWIFT_FE) -c -primary-file $< $(SWIFT_FEFLAGS) \
	  -module-name main -o main.o \
          -emit-module-path main.part.swiftmodule \
	  -import-objc-header $(SRCDIR)/bridging-header.h \
          -Xcc -I$(SRCDIR)/foo.hmap -I$(SRCDIR) 
	$(SWIFT_FE) $(SWIFT_FEFLAGS) \
	  -merge-modules -emit-module main.part.swiftmodule \
	  -parse-as-library -sil-merge-partial-modules \
	  -disable-diagnostic-passes -disable-sil-perf-optzns \
	  -import-objc-header $(SRCDIR)/bridging-header.h \
          -Xcc -I$(SRCDIR)/foo.hmap \
          -module-name main -o main.swiftmodule -I$(SRCDIR)
	$(SWIFTC) -o $@ main.o -Xlinker -add_ast_path -Xlinker main.swiftmodule \
          -lDylib -L.

libDylib: dylib.swift
	$(SWIFTC) -g -Onone $^ -emit-library -module-name Dylib -emit-module \
	  -Xlinker -install_name -Xlinker @executable_path/$@.dylib $(SWIFTFLAGS)

clean::
	rm -rf *.swiftmodule *.swiftdoc *.dSYM *~ lib*.dylib a.out *.o

SWIFTFLAGS += -I. -sdk "$(SDKROOT)"
