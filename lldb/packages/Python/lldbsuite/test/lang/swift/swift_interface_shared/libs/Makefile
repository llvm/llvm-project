LEVEL = ../../../make

DYLIB_NAME := $(BASENAME)
DYLIB_SWIFT_SOURCES := libs/$(BASENAME).swift

# don't use the default swift flags, as we don't want -g for the libraries in
# this test
SWIFTFLAGS=-Onone -Xfrontend -serialize-debugging-options -enable-parseable-module-interface

# don't include the wrapped .swiftmodule on linux to make sure we actually use the .swiftinterface
EXCLUDE_WRAPPED_SWIFTMODULE=1

include $(LEVEL)/Makefile.rules

clean::
	rm -f $(BASENAME).swiftinterface
	rm -f $(CLANG_MODULE_CACHE_DIR)/$(BASENAME)-*.swiftmodule
