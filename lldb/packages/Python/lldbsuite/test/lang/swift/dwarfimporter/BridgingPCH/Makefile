LEVEL = ../../../../make

SWIFT_SOURCES := main.swift

# Hack to force the bridging-header.pch rule to be built before everything else.
a.out: main.o
main.o: -import-objc-header bridging-header.pch
.PHONY: -import-objc-header
bridging-header.pch: bridging-header.h
	$(SWIFT_FE) $(SWIFT_FEFLAGS) $^ -emit-pch -o $@

SWIFTFLAGS_EXTRAS = -I$(BUILDDIR)/include

include $(LEVEL)/Makefile.rules

clean::
	rm -rf bridging-header.pch
