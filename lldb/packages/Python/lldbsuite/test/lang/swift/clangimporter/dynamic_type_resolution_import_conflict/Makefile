LEVEL = ../../../../make
SRCDIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
SWIFT_OBJC_INTEROP := 1

all: a.out

include $(LEVEL)/Makefile.rules

# This testcase causes the scratch context to get destroyed by a
# conflict that is triggered via dynamic type resolution. The two
# swift modules talk Objective-C to the conflicting swift module to
# make the conflict possible.

a.out: main.swift libDylib.dylib libConflict.dylib
	$(SWIFTC) -g -Onone $^ -lDylib -lConflict -L$(shell pwd) -o $@ $(SWIFTFLAGS) -Xcc -I$(SRCDIR)/hidden/Bar -I. -import-objc-header $(SRCDIR)/bridging.h

libDylib.dylib: Library.swift
	$(SWIFTC) -g -Onone $^ -emit-library -module-name Dylib -emit-module -Xlinker -install_name -Xlinker @executable_path/$@ -I. $(SWIFTFLAGS)

libConflict.dylib: Conflict.swift
	$(SWIFTC) -g -Onone $^ -emit-library -module-name Conflict -emit-module -Xlinker -install_name -Xlinker @executable_path/$@ -I. -Xcc -I$(SRCDIR)/hidden/Foo $(SWIFTFLAGS)

clean::
	rm -rf *.swiftmodule *.swiftdoc *.dSYM *~ lib*.dylib a.out *.o
