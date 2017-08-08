LEVEL = ../../../../make

include $(LEVEL)/Makefile.rules

everything: three fooey

three: fooey three.swift
	$(SWIFTCC) -g -Onone -L. three.swift -lfooey -I.

fooey: one.swift two.swift
	$(SWIFTCC) -g -Onone one.swift two.swift -emit-library -module-name fooey -emit-module

cleanup:
	rm -rf *.swiftmodule *.swiftdoc *.dSYM *~ lib*.dylib three