LEVEL = ../../../../make
SWIFT_OBJC_INTEROP := 1

include $(LEVEL)/Makefile.rules

BUILDDIR := $(realpath $(shell pwd))
BOTDIR := $(BUILDDIR)/buildbot
USERDIR := $(BUILDDIR)/user

a.out: main.m libFoo.dylib
	$(CC) $(CFLAGS) $(MANDATORY_MODULE_BUILD_CFLAGS) -I$(shell pwd) -lFoo -L$(shell pwd) -o $@ $<
ifneq "$(CODESIGN)" ""
	$(CODESIGN) -s - "$@"
endif
	mv $(BOTDIR) $(USERDIR)

libFoo.dylib: Foo.swift
	mkdir -p $(BOTDIR)
	cp -r $(SRCDIR)/Foo $(BOTDIR)/
	$(MAKE) MAKE_DSYM=YES CC=$(CC) SWIFTC=$(SWIFTC) \
		ARCH=$(ARCH) DSYMUTIL=$(DSYMUTIL) \
		BASENAME=$(shell basename $< .swift) \
		BOTDIR=$(BOTDIR) \
		VPATH=$(SRCDIR) -I $(SRCDIR) -f $(SRCDIR)/dylib.mk all
	UUID=$$(dwarfdump --uuid "$@.dSYM" | awk '{ print $$2 }') ; \
	echo "UUID = $$UUID" ; \
	PLIST="$@.dSYM/Contents/Resources/$$UUID.plist" ; \
	echo "Generating $$PLIST ..."; \
	echo '<?xml version="1.0" encoding="UTF-8"?>'>$$PLIST; \
	echo '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">'>>$$PLIST; \
	echo '<plist version="1.0">'>>$$PLIST; \
	echo '<dict>'>>$$PLIST; \
	echo '  <key>DBGSourcePathRemapping</key>'>>$$PLIST; \
	echo '  <dict>'>>$$PLIST; \
	echo "    <key>$(BOTDIR)</key>">>$$PLIST; \
	echo "    <string>$(USERDIR)</string>">>$$PLIST; \
	echo '  </dict>'>>$$PLIST; \
	echo '</dict>'>>$$PLIST; \
	echo '</plist>'>>$$PLIST

clean::
	rm -rf $(USERDIR) rm -rf *.swiftmodule *.swiftdoc *.dSYM *~ lib*.dylib a.out *.o
