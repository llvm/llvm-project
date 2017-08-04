LEVEL = ../../../../make

include $(LEVEL)/Makefile.rules

everything: main

main: main.swift
	$(SWIFTCC) -g -Onone -enforce-exclusivity=checked -o main main.swift

cleanup:
	rm -rf main *.dSYM lib*.dylib *~