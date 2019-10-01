LEVEL = ../../../make

SWIFT_SOURCES := main.swift
SWIFTFLAGS_EXTRAS = -framework aTestFramework -framework Cocoa -F $(BUILDDIR) -Xlinker -rpath -Xlinker $(BUILDDIR)

all: aTestFramework.framework/Versions/A/aTestFramework a.out a.out.dSYM

aTestFramework.framework/Versions/A/aTestFramework : aTestFramework.mm
	$(MAKE) MAKE_DSYM=YES CC=$(CC) CXX=$(CXX) \
		ARCH=$(ARCH) DSYMUTIL=$(DSYMUTIL) \
		SDKROOT=$(SDKROOT) \
		SRCDIR=$(SRCDIR) \
		VPATH=$(SRCDIR) -I $(SRCDIR) -f $(SRCDIR)/framework.mk all

clean::
	$(MAKE) -f $(SRCDIR)/framework.mk clean
	rm -rf aTestFramework.framework

include $(LEVEL)/Makefile.rules
