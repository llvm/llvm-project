LEVEL = ../../../../make
SWIFT_SOURCES = main.swift
SWIFT_OBJC_INTEROP := 1
LD_EXTRAS = -ldylib -L$(shell pwd)

# This test builds a .dylib whose .swiftmodule imports a Clang module
# which uses a header file "Foo.h". The main swift program also
# imports a Clang module that imports a header file called "Foo.h",
# but it is in a different directory. We are differen -Xcc -I paths to
# switch between the two versions of "Foo.h" during build time.

all: libdylib.dylib a.out

include $(LEVEL)/Makefile.rules
SWIFTFLAGS += -Xcc -I$(SRCDIR)/Bar -I. -Xcc -I$(SRCDIR)/Foo

libdylib.dylib: dylib.swift
	$(MAKE) MAKE_DSYM=YES X=$(MAKE_DSYM) CC=$(CC) SWIFTC=$(SWIFTC) \
		ARCH=$(ARCH) DSYMUTIL=$(DSYMUTIL) \
		DYLIB_NAME=$(shell basename $< .swift) \
		VPATH=$(SRCDIR) -I $(SRCDIR) -f $(SRCDIR)/dylib.mk all

clean::
	rm -rf *.swiftmodule *.swiftdoc *.dSYM *~ lib*.dylib a.out *.o
