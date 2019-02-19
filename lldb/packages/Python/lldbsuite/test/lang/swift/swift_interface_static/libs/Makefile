LEVEL = ../../../make

# Set the dylib variables so the rules for generating swift objects and
# modules are active.
DYLIB_NAME := $(BASENAME)
DYLIB_SWIFT_SOURCES := libs/$(BASENAME).swift

# Don't use the default swift flags, as we don't want -g for the libraries in
# this test.
SWIFTFLAGS=-Onone -Xfrontend -serialize-debugging-options -enable-parseable-module-interface -parse-as-library

# Don't include the wrapped .swiftmodule on Linux to make sure we use
# the .swiftinterface. The issue here is that if the .swiftmodule is wrapped
# and linked into the library, LLDB may find it there and load it before the
# .swiftinterface loading path can be exercised.
EXCLUDE_WRAPPED_SWIFTMODULE=1

# Activates the rules for generating a static lib based on the .o files
# corresponding to the dylib swift sources above.
ARCHIVE_NAME := lib$(BASENAME).a
ARCHIVE_OBJECTS = $(strip $(DYLIB_SWIFT_SOURCES:.swift=.o))

$(ARCHIVE_NAME): $(BASENAME).swiftmodule

include $(LEVEL)/Makefile.rules

# Generate only the static lib (all would generate a dylib as well).
static_only: $(ARCHIVE_NAME)

clean::
	rm -f $(BASENAME).swiftinterface
