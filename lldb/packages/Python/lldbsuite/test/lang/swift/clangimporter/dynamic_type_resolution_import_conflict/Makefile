LEVEL = ../../../../make
SWIFT_SOURCES=main.swift
SWIFT_OBJC_INTEROP := 1
LD_EXTRAS=-lDylib -lConflict -L$(shell pwd)
SWIFTFLAGS_EXTRAS= -Xcc -I$(SRCDIR)/hidden/Bar -I. -import-objc-header $(SRCDIR)/bridging.h

all: libDylib.dylib libConflict.dylib a.out

include $(LEVEL)/Makefile.rules

# This testcase causes the scratch context to get destroyed by a
# conflict that is triggered via dynamic type resolution. The two
# swift modules talk Objective-C to the conflicting swift module to
# make the conflict possible.

libDylib.dylib: Dylib.swift
	$(MAKE) MAKE_DSYM=YES X=$(MAKE_DSYM) CC=$(CC) SWIFTC=$(SWIFTC) \
		ARCH=$(ARCH) DSYMUTIL=$(DSYMUTIL) \
		BASENAME=$(shell basename $< .swift) \
		VPATH=$(SRCDIR) -I $(SRCDIR) -f $(SRCDIR)/dylib.mk all

libConflict.dylib: Conflict.swift
	$(MAKE) MAKE_DSYM=YES X=$(MAKE_DSYM) CC=$(CC) SWIFTC=$(SWIFTC) \
		ARCH=$(ARCH) DSYMUTIL=$(DSYMUTIL) \
		BASENAME=$(shell basename $< .swift) \
		SWIFTFLAGS_EXTRAS="-I. -Xcc -I$(SRCDIR)/hidden/Foo" \
		VPATH=$(SRCDIR) -I $(SRCDIR) -f $(SRCDIR)/dylib.mk all

clean::
	rm -rf *.swiftmodule *.swiftdoc *.dSYM *~ lib*.dylib a.out *.o
