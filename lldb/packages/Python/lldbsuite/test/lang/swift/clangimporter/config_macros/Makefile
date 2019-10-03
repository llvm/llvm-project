LEVEL = ../../../../make
SWIFT_OBJC_INTEROP := 1
SWIFT_SOURCES := main.swift
SWIFTFLAGS_EXTRAS := -Xcc -DDEBUG=1 -Xcc -DSPACE -Xcc -UNDEBUG -I. -L. -lDylib

all: libDylib.dylib a.out

include $(LEVEL)/Makefile.rules

libDylib.dylib: dylib.swift
	$(MAKE) MAKE_DSYM=$(MAKE_DSYM) CC=$(CC) SWIFTC=$(SWIFTC) \
		ARCH=$(ARCH) DSYMUTIL=$(DSYMUTIL) \
		BASENAME=$(shell basename $< .swift) \
		VPATH=$(SRCDIR) -I $(SRCDIR) -f $(SRCDIR)/dylib.mk all

clean::
	rm -rf *.swiftmodule *.swiftdoc *.dSYM *~ lib*.dylib a.out *.o

