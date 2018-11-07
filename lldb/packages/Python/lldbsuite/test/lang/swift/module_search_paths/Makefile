LEVEL = ../../../make

SWIFT_SOURCES := main.swift

include $(LEVEL)/Makefile.rules

module: module.swift
	$(SWIFTC) $(SWIFT_MODULE_CACHE_FLAGS) -g -Onone -emit-module -module-name Module $<
ifneq "$(CODESIGN)" ""
	$(CODESIGN) -s - "$@"
endif

cleanup:
	rm -rf *.swiftdoc *.swiftmodule
