LEVEL = ../../make
SRCDIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

include $(LEVEL)/Makefile.rules

a.out: main.swift libCppLib.dylib
	$(SWIFTC) $(SWIFTFLAGS) -I$(SRCDIR)/CppLib -lCppLib -L$(shell pwd) -o $@ $<

libCppLib.dylib: CppLib/CppLib.cpp
	mkdir -p CppLib
	$(MAKE) VPATH=$(VPATH) -f $(SRCDIR)/CppLib/Makefile

clean::
	rm -rf *.swiftmodule *.swiftdoc *.dSYM *~ lib*.dylib a.out *.o
