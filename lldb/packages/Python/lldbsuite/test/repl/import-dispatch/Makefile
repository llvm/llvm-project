LEVEL = ../../make
SWIFT_OBJC_INTEROP := 1
LLDB=$(shell echo $(LLDB_LIB_DIR)/lldb)
# This Makefile overwrites the default rules.
EXE := a.out

first: a.out

include $(LEVEL)/Makefile.rules

a.out: input.swift
	cat $< | $(LLDB) --repl="$(CFLAGS_EXTRAS)"
	echo "libdispatch could be imported successfully!">$@

input.swift:
	echo "import Dispatch" >$@

clean::
	rm -rf a.out
