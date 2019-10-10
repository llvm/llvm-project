EXE = PlaygroundStub
SWIFT_SOURCES = PlaygroundStub.swift

# The setup here is complex. We want to build the file with a deployment target
# earlier than the availability set in the source file. This is made slightly
# more complex by watchOS having misaligned version numbers.
ifeq ($(TRIPLE),)
TRIPLE := x86_64-apple-macosx10.10
else
override TRIPLE := $(shell echo $(TRIPLE) | sed 's/os[0-9.]*/os7.0/' | sed 's/watchos7.0/watchos5.0/')
endif

# The deployment target we set is pre-ABI stability. The Swift driver will not
# point the RPATH at the system library. Do it manually.
LD_EXTRAS := -Xlinker -rpath -Xlinker /usr/lib/swift

PlaygroundStub: libPlaygroundsRuntime.dylib

include Makefile.rules

libPlaygroundsRuntime.dylib: PlaygroundsRuntime.swift
	$(MAKE) -f $(MAKEFILE_RULES) \
		DYLIB_SWIFT_SOURCES=PlaygroundsRuntime.swift \
		DYLIB_NAME=PlaygroundsRuntime

