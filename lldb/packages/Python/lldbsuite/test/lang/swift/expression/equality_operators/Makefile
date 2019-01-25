LEVEL = ../../../../make

EXE=three
SWIFT_SOURCES=three.swift
LD_EXTRAS=-L$(shell pwd) -lfooey
SWIFTFLAGS_EXTRAS= -I$(shell pwd)

all: libfooey.dylib three

include $(LEVEL)/Makefile.rules

libfooey.dylib: one.swift two.swift
	$(SWIFTC) $(SWIFT_MODULE_CACHE_FLAGS) -g -Onone $^ -emit-library -module-name fooey -emit-module -Xlinker -install_name -Xlinker @executable_path/$@
ifneq "$(CODESIGN)" ""#
	$(CODESIGN) -s - "$@"
endif

clean::
	rm -rf *.swiftmodule *.swiftdoc *.dSYM *~ lib*.dylib three
