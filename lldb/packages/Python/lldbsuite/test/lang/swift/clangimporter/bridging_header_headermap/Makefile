LEVEL = ../../../../make
SWIFT_OBJC_INTEROP := 1
SWIFT_SOURCES := main.swift
MAKE_DSYM := NO
LD_EXTRAS := -ldylib -L.
SWIFTFLAGS_EXTRAS = -import-objc-header $(SRCDIR)/bridging-header.h \
                    -Xcc -I$(SRCDIR)/foo.hmap -I. -I$(SRCDIR)

all: libdylib.dylib a.out

# This test builds a .dylib whose .swiftmodule imports a Clang module
# which uses a header file "Foo.h". The main swift program also
# imports a Clang module that imports a header file called "Foo.h",
# but it is in a different directory. We are using a headermap to
# switch between the two versions of "Foo.h" during build time.

include $(LEVEL)/Makefile.rules

libdylib.dylib: dylib.swift
	$(MAKE) MAKE_DSYM=YES CC=$(CC) SWIFTC=$(SWIFTC) \
		ARCH=$(ARCH) DSYMUTIL=$(DSYMUTIL) \
		DYLIB_NAME=$(shell basename $< .swift) \
		VPATH=$(SRCDIR) -I $(SRCDIR) -f $(SRCDIR)/dylib.mk all

clean::
	rm -rf *.swiftmodule *.swiftdoc *.dSYM *~ lib*.dylib a.out *.o

