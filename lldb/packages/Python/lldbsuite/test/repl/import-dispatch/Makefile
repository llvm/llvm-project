LEVEL = ../../make
SWIFT_OBJC_INTEROP := 1
# This Makefile overwrites the default rules.
EXE=None

first: a.out

include $(LEVEL)/Makefile.rules

a.out: input.swift
	cat $< | lldb --repl "$(CFLAGS_EXTRAS)"
	echo "libdispatch could be imported successfully!">$@

input.swift:
	echo "import Dispatch" >$@

clean::
	rm -rf a.out
