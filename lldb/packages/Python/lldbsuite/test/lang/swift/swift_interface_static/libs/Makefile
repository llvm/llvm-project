LEVEL = ../../../make

# set the dylib variables so the rules for generating swift objects and
# modules are active
DYLIB_NAME := $(BASENAME)
DYLIB_SWIFT_SOURCES := libs/$(BASENAME).swift

# don't use the default swift flags, as we don't want -g for the libraries in
# this test
SWIFTFLAGS=-Onone -Xfrontend -serialize-debugging-options -enable-parseable-module-interface -parse-as-library

# don't include the wrapped .swiftmodule on linux
EXCLUDE_WRAPPED_SWIFTMODULE=1

# activates the rules for generating a static lib based on the .o files
# corresponding to the dylib swift sources above
ARCHIVE_NAME := lib$(BASENAME).a
ARCHIVE_OBJECTS = $(strip $(DYLIB_SWIFT_SOURCES:.swift=.o))

$(ARCHIVE_NAME): $(BASENAME).swiftmodule

include $(LEVEL)/Makefile.rules

# on Linux we need to wrap the swiftmodule as a .o and include it too
ifneq "$(OS)" "Darwin"
ARCHIVE_OBJECTS += $(BASENAME).swiftmodule.o
endif

# generate only the static lib (all would generate a dylib as well)
static_only: $(ARCHIVE_NAME)

clean::
	rm -f $(BASENAME).swiftinterface
