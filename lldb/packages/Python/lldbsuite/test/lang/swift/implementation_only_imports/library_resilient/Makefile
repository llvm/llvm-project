LEVEL = ../../../../make

SWIFT_SOURCES=main.swift
SWIFTFLAGS_EXTRAS= -I.
LD_EXTRAS=-L.

all: SomeLibraryCore.swiftmodule SomeLibrary.swiftmodule a.out

include $(LEVEL)/Makefile.rules

SomeLibraryCore.swiftmodule: SomeLibraryCore.swift
	$(MAKE) BASENAME=$(shell basename $@ .swiftmodule) \
		SWIFTFLAGS_EXTRAS=-enable-library-evolution \
		VPATH=$(SRCDIR) -f $(SRCDIR)/../dylib.mk all

SomeLibrary.swiftmodule: SomeLibrary.swift SomeLibraryCore.swiftmodule
	$(MAKE) BASENAME=$(shell basename $@ .swiftmodule) \
		SWIFTFLAGS_EXTRAS=-enable-library-evolution \
		VPATH=$(SRCDIR) -f $(SRCDIR)/../dylib.mk all

clean::
	rm -rf *.o *.dylib *.so *.dll *.dSYM *.swiftdoc *.swiftmodule *.swiftinterface

