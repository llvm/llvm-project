LEVEL = ../../../make

SWIFT_SOURCES := main.swift

include $(LEVEL)/Makefile.rules

module:
	$(SWIFTCC) -g -Onone -emit-module -module-name Module module.swift

cleanup:
	rm -rf *.swiftdoc *.swiftmodule
