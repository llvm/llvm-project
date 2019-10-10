SWIFT_SOURCES := main.swift
SWIFTFLAGS_EXTRAS = -framework aTestFramework -framework CoreGraphics -F $(BUILDDIR) -Xlinker -rpath -Xlinker $(BUILDDIR)

all: aTestFramework.framework/Versions/A/aTestFramework a.out a.out.dSYM

include Makefile.rules

aTestFramework.framework/Versions/A/aTestFramework : aTestFramework.mm
	$(MAKE) -f $(MAKEFILE_RULES) \
		MAKE_DSYM=YES DYLIB_ONLY=YES \
		FRAMEWORK=aTestFramework DYLIB_CXX_SOURCES=aTestFramework.mm \
		FRAMEWORK_HEADERS=$(SRCDIR)/aTestFramework.h \
		FRAMEWORK_MODULES=$(SRCDIR)/module.modulemap \
		LD_EXTRAS="-framework Foundation"

