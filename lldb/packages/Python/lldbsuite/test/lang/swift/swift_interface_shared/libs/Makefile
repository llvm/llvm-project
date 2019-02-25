LEVEL = ../../../make

DYLIB_NAME := $(BASENAME)
DYLIB_SWIFT_SOURCES := libs/$(BASENAME).swift

# Don't use the default swift flags, as we don't want -g for the libraries in
# this test.
SWIFTFLAGS=-Onone -Xfrontend -serialize-debugging-options -enable-parseable-module-interface

# Don't include the wrapped .swiftmodule on Linux to make sure we actually use
# the .swiftinterface. The issue here is that if the .swiftmodule is wrapped
# and linked into the dylib, LLDB will find it there and load it before the
# .swiftinterface loading path can be exercised.
EXCLUDE_WRAPPED_SWIFTMODULE=1

include $(LEVEL)/Makefile.rules

clean::
	rm -f $(BASENAME).swiftinterface
	rm -f $(CLANG_MODULE_CACHE_DIR)/$(BASENAME)-*.swiftmodule
