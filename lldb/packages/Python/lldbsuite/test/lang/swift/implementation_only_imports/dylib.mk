# There's one extra level here because this is called from the tests in the subdirectories.
DYLIB_ONLY := YES
DYLIB_NAME := $(BASENAME)
DYLIB_SWIFT_SOURCES := $(DYLIB_NAME).swift
# Disable debug info for the library.
override MAKE_DSYM := NO
EXCLUDE_WRAPPED_SWIFTMODULE := YES
SWIFTFLAGS := -gnone -I. -module-link-name $(BASENAME)
LD_EXTRAS := -L.

include Makefile.rules
