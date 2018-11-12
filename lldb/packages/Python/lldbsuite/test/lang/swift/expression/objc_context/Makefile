LEVEL = ../../../../make
SRCDIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
SWIFT_OBJC_INTEROP := 1

# This test builds an Objective-C main program that imports two Swift
# .dylibs with conflicting ClangImporter search paths.

include $(LEVEL)/Makefile.rules

a.out: main.m libFoo.dylib
	$(CC) $(CFLAGS) $(MANDATORY_MODULE_BUILD_CFLAGS) -I$(shell pwd) -lFoo -L$(shell pwd) -o $@ $<
ifneq "$(CODESIGN)" ""
	$(CODESIGN) -s - "$@"
endif

lib%.dylib: %.swift
	$(SWIFTC) -g -Onone $^ -emit-library -module-name $(shell basename $< .swift) -emit-module -Xlinker -install_name -Xlinker @executable_path/$@ $(SWIFTFLAGS) -emit-objc-header-path $(shell basename $< .swift).h
ifneq "$(CODESIGN)" ""
	$(CODESIGN) -s - "$@"
endif

clean::
	rm -rf *.swiftmodule *.swiftdoc *.dSYM *~ lib*.dylib a.out *.o
