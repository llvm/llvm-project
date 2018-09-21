LEVEL = ../../../../make
SRCDIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
SWIFT_OBJC_INTEROP := 1

all: a.out

include $(LEVEL)/Makefile.rules

.PHONY: libLibrary

libLibrary: Library.swift
	$(SWIFT_FE) -c -primary-file $^ $(SWIFT_FEFLAGS) \
          -emit-module-path Library.part.swiftmodule \
          -parse-as-library -module-name Library -o Library.o -I.
	$(SWIFT_FE) $(SWIFT_FEFLAGS) -merge-modules -emit-module \
          -parse-as-library -sil-merge-partial-modules \
          -disable-diagnostic-passes -disable-sil-perf-optzns \
          -module-name Library Library.part.swiftmodule -o Library.swiftmodule \
	  -I$(shell pwd)
	$(SWIFTC) -Xlinker -dylib -o $@.dylib Library.o \
	  -Xlinker -add_ast_path -Xlinker Library.swiftmodule \
	  -Xlinker -install_name -Xlinker @executable_path/libLibrary.dylib

a.out: main.swift libLibrary
	$(SWIFT_FE) -c -primary-file $< $(SWIFT_FEFLAGS) \
	  -module-name main -o main.o \
          -emit-module-path main.part.swiftmodule \
          -import-objc-header $(VPATH)/BridgingHeader.h \
          -I. -Xcc -DSYNTAX_ERROR=1
	$(SWIFT_FE) $(SWIFT_FEFLAGS) \
	  -merge-modules -emit-module main.part.swiftmodule \
	  -parse-as-library -sil-merge-partial-modules \
	  -disable-diagnostic-passes -disable-sil-perf-optzns \
          -import-objc-header $(VPATH)/BridgingHeader.h \
          -I$(shell pwd) -Xcc -DSYNTAX_ERROR=1 -sdk $(SDKROOT) \
          -module-name main -o main.swiftmodule
	$(SWIFTC) -o $@ main.o -Xlinker -add_ast_path -Xlinker main.swiftmodule \
          -lLibrary -L.

clean::
	rm -rf *.swiftmodule *.swiftdoc *.dSYM *~ lib*.dylib a.out *.o


SWIFTFLAGS += -sdk "$(SDKROOT)"
