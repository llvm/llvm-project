##
# libc++ Makefile
##

SRCDIRS = .
DESTDIR = $(DSTROOT)

OBJROOT=.
SYMROOT=.
export TRIPLE=-apple-

ifeq (,$(RC_INDIGO))
	INSTALL_PREFIX=""
else
	INSTALL_PREFIX="$(SDKROOT)"
endif
INSTALL_DIR=$(DSTROOT)/$(INSTALL_PREFIX)

.PHONY: help installsrc clean installheaders install

help::
	@echo "Use make install DSTROOT=<destination>"

installsrc:: $(SRCROOT)

	ditto $(SRCDIRS)/include $(SRCROOT)/include
	ditto $(SRCDIRS)/lib $(SRCROOT)/lib
	ditto $(SRCDIRS)/src $(SRCROOT)/src
	ditto $(SRCDIRS)/Makefile $(SRCROOT)/Makefile

clean::

# The installheaders target is used by clang's runtime/libcxx makefile.
installheaders::
	mkdir -p $(HEADER_DIR)/c++/v1/ext
	(cd $(SRCDIRS)/include && tar cf - --exclude=support .) | \
	  (cd $(HEADER_DIR)/c++/v1 && tar xf -)
	chmod 755 $(HEADER_DIR)/c++/v1
	chmod 644 $(HEADER_DIR)/c++/v1/*
	chmod 755 $(HEADER_DIR)/c++/v1/ext
	chmod 644 $(HEADER_DIR)/c++/v1/ext/*

install::

	cd lib && ./buildit
	ditto lib/libc++.1.dylib $(SYMROOT)/usr/lib/libc++.1.dylib
	cd lib && dsymutil -o $(SYMROOT)/libc++.1.dylib.dSYM \
	  $(SYMROOT)/usr/lib/libc++.1.dylib
	mkdir -p $(INSTALL_DIR)/usr/lib
	strip -S -o $(INSTALL_DIR)/usr/lib/libc++.1.dylib \
	  $(SYMROOT)/usr/lib/libc++.1.dylib
	cd $(INSTALL_DIR)/usr/lib && ln -s libc++.1.dylib libc++.dylib
