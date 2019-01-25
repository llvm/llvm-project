LEVEL = ../../../../make
SWIFT_OBJC_INTEROP := 1
# This Makefile overwrites the default rules.
USESWIFTDRIVER := 0
EXE := a.out

all: a.out

# This test builds an Objective-C main program that imports two Swift
# .dylibs with conflicting ClangImporter search paths.

include $(LEVEL)/Makefile.rules

a.out: main.o
	$(SWIFTC) $(SWIFTFLAGS) $< -lFoo -lBar -L$(shell pwd) -o $@ \
	    -Xlinker -add_ast_path -Xlinker $(shell pwd)/Foo.swiftmodule \
	    -Xlinker -add_ast_path -Xlinker $(shell pwd)/Bar.swiftmodule
ifneq "$(CODESIGN)" ""
	$(CODESIGN) -s - "$@"
endif

main.o: main.m libFoo.a libBar.a
	$(CC) $(CFLAGS) $(MANDATORY_MODULE_BUILD_CFLAGS) -c -o $@ $< \
		-I$(shell pwd)

lib%.o: %.swift
	$(SWIFTC) $(SWIFTFLAGS) $^ -c -parse-as-library -o $@ \
		-Xcc -I$(SRCDIR) \
		-Xcc -I$(SRCDIR)/objcmodules/$(shell basename $< .swift) \
		-module-name $*
	$(SWIFTC) $(SWIFTFLAGS) $^ -emit-module -parse-as-library \
		-emit-objc-header-path $(shell basename $< .swift).h \
		-Xcc -I$(SRCDIR) \
		-Xcc -I$(SRCDIR)/objcmodules/$(shell basename $< .swift) \
		-module-name $*

lib%.a: lib%.o
	$(AR) $(ARFLAGS) $@ -L$(shell pwd) $<

clean::
	rm -rf *.swiftmodule *.swiftdoc *.dSYM *~ lib*.a a.out *.o
