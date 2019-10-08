EXE := three
SWIFT_SOURCES := three.swift
LD_EXTRAS := -L. -lfooey
SWIFTFLAGS_EXTRAS := -I.

all: libfooey.dylib three

include Makefile.rules

libfooey.dylib: one.swift two.swift
	$(MAKE) -f $(MAKEFILE_RULES) \
		MAKE_DSYM=YES DYLIB_ONLY=YES DYLIB_NAME=fooey \
		SWIFT_SOURCES="one.swift two.swift" \
		SWIFTFLAGS_EXTRAS=-I$(BUILDDIR)
