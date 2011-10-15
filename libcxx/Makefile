##
# libcpp Makefile
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

help::
	echo Use make install DSTROOT=<destination>

installsrc:: $(SRCROOT)

	ditto $(SRCDIRS)/include $(SRCROOT)/include
	ditto $(SRCDIRS)/lib $(SRCROOT)/lib
	ditto $(SRCDIRS)/src $(SRCROOT)/src
	ditto $(SRCDIRS)/Makefile $(SRCROOT)/Makefile

clean::

installhdrs::

	mkdir -p $(DSTROOT)/$(INSTALL_PREFIX)/usr/lib/c++/v1/ext
	mkdir -p $(DSTROOT)/$(INSTALL_PREFIX)/usr/clang-ide/lib/c++/v1/ext
	mkdir -p $(DSTROOT)/$(INSTALL_PREFIX)/Developer/usr/lib/c++/v1/ext
	mkdir -p $(DSTROOT)/$(INSTALL_PREFIX)/Developer/Platforms/iPhoneOS.platform/usr/lib/c++/v1/ext
	rsync -r --exclude=".*" $(SRCDIRS)/include/* $(DSTROOT)/$(INSTALL_PREFIX)/usr/lib/c++/v1/
	rsync -r --exclude=".*" $(SRCDIRS)/include/* $(DSTROOT)/$(INSTALL_PREFIX)/usr/clang-ide/lib/c++/v1/
	rsync -r --exclude=".*" $(SRCDIRS)/include/* $(DSTROOT)/$(INSTALL_PREFIX)/Developer/usr/lib/c++/v1/
	rsync -r --exclude=".*" $(SRCDIRS)/include/* $(DSTROOT)/$(INSTALL_PREFIX)/Developer/Platforms/iPhoneOS.platform/usr/lib/c++/v1/
	chown -R root:wheel $(DSTROOT)/$(INSTALL_PREFIX)/usr/lib/c++
	chown -R root:wheel $(DSTROOT)/$(INSTALL_PREFIX)/usr/clang-ide/lib/c++
	chown -R root:wheel $(DSTROOT)/$(INSTALL_PREFIX)/Developer/usr/lib/c++
	chown -R root:wheel $(DSTROOT)/$(INSTALL_PREFIX)/Developer/Platforms/iPhoneOS.platform/usr/lib/c++
	chmod 755 $(DSTROOT)/$(INSTALL_PREFIX)/usr/lib/c++/v1
	chmod 755 $(DSTROOT)/$(INSTALL_PREFIX)/usr/clang-ide/lib/c++/v1
	chmod 755 $(DSTROOT)/$(INSTALL_PREFIX)/Developer/usr/lib/c++/v1
	chmod 755 $(DSTROOT)/$(INSTALL_PREFIX)/Developer/Platforms/iPhoneOS.platform/usr/lib/c++/v1
	chmod 644 $(DSTROOT)/$(INSTALL_PREFIX)/usr/lib/c++/v1/*
	chmod 644 $(DSTROOT)/$(INSTALL_PREFIX)/usr/clang-ide/lib/c++/v1/*
	chmod 644 $(DSTROOT)/$(INSTALL_PREFIX)/Developer/usr/lib/c++/v1/*
	chmod 644 $(DSTROOT)/$(INSTALL_PREFIX)/Developer/Platforms/iPhoneOS.platform/usr/lib/c++/v1/*
	chmod 755 $(DSTROOT)/$(INSTALL_PREFIX)/usr/lib/c++/v1/ext
	chmod 755 $(DSTROOT)/$(INSTALL_PREFIX)/usr/clang-ide/lib/c++/v1/ext
	chmod 755 $(DSTROOT)/$(INSTALL_PREFIX)/Developer/usr/lib/c++/v1/ext
	chmod 755 $(DSTROOT)/$(INSTALL_PREFIX)/Developer/Platforms/iPhoneOS.platform/usr/lib/c++/v1/ext
	chmod 644 $(DSTROOT)/$(INSTALL_PREFIX)/usr/lib/c++/v1/ext/*
	chmod 644 $(DSTROOT)/$(INSTALL_PREFIX)/usr/clang-ide/lib/c++/v1/ext/*
	chmod 644 $(DSTROOT)/$(INSTALL_PREFIX)/Developer/usr/lib/c++/v1/ext/*
	chmod 644 $(DSTROOT)/$(INSTALL_PREFIX)/Developer/Platforms/iPhoneOS.platform/usr/lib/c++/v1/ext/*

install:: installhdrs $(DESTDIR)

	cd lib && ./buildit
	ditto lib/libc++.1.dylib $(SYMROOT)/usr/lib/libc++.1.dylib
	cd lib && dsymutil -o $(SYMROOT)/libc++.1.dylib.dSYM  $(SYMROOT)/usr/lib/libc++.1.dylib
	mkdir -p $(DSTROOT)/$(INSTALL_PREFIX)/usr/lib
	strip -S -o $(DSTROOT)/$(INSTALL_PREFIX)/usr/lib/libc++.1.dylib $(SYMROOT)/usr/lib/libc++.1.dylib
	cd $(DSTROOT)/$(INSTALL_PREFIX)/usr/lib && ln -s libc++.1.dylib libc++.dylib
