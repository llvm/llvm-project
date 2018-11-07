LEVEL = ../../make
SRCDIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
CURDIR = $(shell pwd)

include $(LEVEL)/Makefile.rules

a.out: main.swift libCppLib.dylib libWrapper.dylib
	$(SWIFTC) $(SWIFTFLAGS) -Xlinker -rpath -Xlinker $(CURDIR) -I$(CURDIR) -lWrapper -L$(CURDIR) -o $@ $<
ifneq "$(CODESIGN)" ""
	$(CODESIGN) -s - "$@"
endif


libCppLib.dylib: CppLib/CppLib.cpp
	mkdir -p CppLib
	$(MAKE) VPATH=$(VPATH) -f $(SRCDIR)/CppLib/Makefile
	install_name_tool -id $@ $@

libWrapper.dylib: wrapper.swift libCppLib.dylib
	$(SWIFTC) $(SWIFTFLAGS) -emit-module -emit-library -module-link-name Wrapper -module-name Wrapper -o $@ -L$(CURDIR) -lCppLib -I$(SRCDIR)/CppLib -o $@ -Xlinker -install_name -Xlinker @executable_path/$@ $<
ifneq "$(CODESIGN)" ""
	$(CODESIGN) -s - "$@"
endif

clean::
	rm -rf *.swiftmodule *.swiftdoc *.dSYM *~ lib*.dylib a.out *.o
