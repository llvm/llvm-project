LEVEL = ../../../../make
SRCDIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
SWIFT_OBJC_INTEROP := 1
SWIFT_SOURCES = Foo.swift Bar.swift

# This test builds an Objective-C main program that imports two Swift
# .dylibs with conflicting ClangImporter search paths.

include $(LEVEL)/Makefile.rules

a.out: main.o libFoo.a libBar.a
	$(SWIFTC) $(SWIFTFLAGS) $< -lFoo -lBar -L$(shell pwd) -o $@ \
	    -Xlinker -add_ast_path -Xlinker $(shell pwd)/Foo.swiftmodule \
	    -Xlinker -add_ast_path -Xlinker $(shell pwd)/Bar.swiftmodule

main.o: main.m Foo.o Bar.o
	$(CC) $(CFLAGS) $(MANDATORY_MODULE_BUILD_CFLAGS) -c -o $@ $< \
		-I$(shell pwd)

%.o: %.swift
	$(SWIFTC) $(SWIFTFLAGS) $^ -c -parse-as-library -o $@ \
		-Xcc -I$(SRCDIR) \
		-Xcc -I$(SRCDIR)/objcmodules/$(shell basename $< .swift)
	$(SWIFTC) $(SWIFTFLAGS) $^ -emit-module -parse-as-library \
		-emit-objc-header-path $(shell basename $< .swift).h \
		-Xcc -I$(SRCDIR) \
		-Xcc -I$(SRCDIR)/objcmodules/$(shell basename $< .swift)

lib%.a: %.o
	$(AR) $(ARFLAGS) $@ -L$(shell pwd) $<

clean::
	rm -rf *.swiftmodule *.swiftdoc *.dSYM *~ lib*.a a.out *.o
