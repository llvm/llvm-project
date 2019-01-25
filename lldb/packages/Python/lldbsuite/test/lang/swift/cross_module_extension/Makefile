LEVEL = ../../../make

EXE=main
SWIFT_SOURCES=main.swift
LD_EXTRAS=-lmoda -lmodb -L$(shell pwd)
SWIFTFLAGS_EXTRAS=-I$(shell pwd)

all: libmoda.dylib libmodb.dylib main

include $(LEVEL)/Makefile.rules


libmoda.dylib: moda.swift
	$(MAKE) MAKE_DSYM=YES CC=$(CC) SWIFTC=$(SWIFTC) \
		ARCH=$(ARCH) DSYMUTIL=$(DSYMUTIL) \
		BASENAME=$(shell basename $< .swift) \
		VPATH=$(SRCDIR) -I $(SRCDIR) -f $(SRCDIR)/dylib.mk all


libmodb.dylib: modb.swift
	$(MAKE) MAKE_DSYM=YES CC=$(CC) SWIFTC=$(SWIFTC) \
		ARCH=$(ARCH) DSYMUTIL=$(DSYMUTIL) \
		BASENAME=$(shell basename $< .swift) \
		LD_EXTRAS="-lmoda -L$(shell pwd)" \
		VPATH=$(SRCDIR) -I $(SRCDIR) -f $(SRCDIR)/dylib.mk all

clean::
	rm -rf a.out.dSYM a.out libmoda.dylib libmoda.dylib.dSYM libmodb.dylib libmodb.dylib.dSYM moda.swiftdoc moda.swiftmodule modb.swiftmodule modb.swiftdoc

