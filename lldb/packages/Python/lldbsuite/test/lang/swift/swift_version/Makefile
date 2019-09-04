LEVEL = ../../../make

EXE=main
SWIFT_SOURCES=main.swift
SWIFTFLAGS_EXTRAS= -I$(shell pwd)
LD_EXTRAS=-L$(shell pwd) -lmod5 -lmod4

all: libmod5.dylib libmod4.dylib main

include $(LEVEL)/Makefile.rules

MOD5_FLAGS = -swift-version 5
MOD4_FLAGS = -swift-version 4

libmod%.dylib: mod%.swift
	$(MAKE) MAKE_DSYM=YES CC=$(CC) SWIFTC=$(SWIFTC) \
		SWIFTFLAGS_EXTRAS="$(MOD$*_FLAGS)" \
		ARCH=$(ARCH) DSYMUTIL=$(DSYMUTIL) \
		BASENAME=$(shell basename $< .swift) \
		VPATH=$(SRCDIR) -I $(SRCDIR) -f $(SRCDIR)/dylib.mk all

clean::
	rm -rf main main.a main.b *.dylib *.dSYM *.swiftdoc *.swiftmodule mod.swift

