LEVEL = ../../make
SWIFT_SOURCES := input.swift
SWIFT_OBJC_INTEROP := 1

.PHONY a.out: input.swift
	cat $< | lldb --repl "$(CFLAGS_EXTRAS)"
	echo "libdispatch could be imported successfully!">$@

input.swift:
	echo "import Dispatch" >$@


include $(LEVEL)/Makefile.rules
