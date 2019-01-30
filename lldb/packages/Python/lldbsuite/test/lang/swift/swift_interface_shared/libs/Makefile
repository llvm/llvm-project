LEVEL = ../../../make

DYLIB_NAME := $(BASENAME)
DYLIB_SWIFT_SOURCES := libs/$(BASENAME).swift

# don't use the default swift flags, as we don't want -g for the libraries in
# this test
SWIFTFLAGS=-Onone -Xfrontend -serialize-debugging-options -enable-parseable-module-interface

include $(LEVEL)/Makefile.rules

clean::
	rm -rf $(BASENAME).swiftinterface
