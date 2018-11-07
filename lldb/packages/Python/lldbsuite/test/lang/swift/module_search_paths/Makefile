LEVEL = ../../../make

SWIFT_SOURCES := main.swift

.PHONY: all module

all: module

include $(LEVEL)/Makefile.rules

# This is an extra dependency that does NOT get linked into the EXE.
module: module.swift
	$(SWIFTC) $(SWIFT_MODULE_CACHE_FLAGS) -g -Onone -emit-module -module-name Module $<
ifneq "$(CODESIGN)" ""
	$(CODESIGN) -s - "$@"
endif

cleanup:
	rm -rf *.swiftdoc *.swiftmodule
