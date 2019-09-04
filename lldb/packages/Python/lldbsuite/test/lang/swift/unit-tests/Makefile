LEVEL = ../../../make

all: dylib xctest
C_SOURCES=xctest.c
EXE=xctest

include $(LEVEL)/Makefile.rules

.PHONY: dylib
dylib:
	$(MAKE) MAKE_DSYM=$(MAKE_DSYM) CC=$(CC) SWIFTC=$(SWIFTC) \
		ARCH=$(ARCH) DSYMUTIL=$(DSYMUTIL) \
		VPATH=$(SRCDIR) -I $(SRCDIR) -f $(SRCDIR)/dylib.mk all

clean::
	rm -rf *.o *.dSYM *.dylib *.swiftdoc *.swiftmodule *.xctest xctest
	$(MAKE) VPATH=$(SRCDIR) -f $(SRCDIR)/dylib.mk clean

