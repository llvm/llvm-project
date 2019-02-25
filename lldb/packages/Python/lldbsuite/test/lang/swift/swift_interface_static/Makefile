LEVEL := ../../../make

EXE := main
SWIFT_SOURCES := main.swift
LD_EXTRAS = -L$(BUILDDIR) -lAA -lBB -lCC
SWIFTFLAGS_EXTRAS = -enable-parseable-module-interface -I$(BUILDDIR)
SWIFT_MODULE_CACHE_FLAGS = -module-cache-path MCP

# setup `all` to run each of the below before building the main executable
all: setup staticA staticB staticC clear_modules

include $(LEVEL)/Makefile.rules

setup:
	# copy the source files into the build directory because we delete them
	# to be really sure we only have/use the .dylibs and .swiftinterfaces
	mkdir -p libs
	cp $(SRCDIR)/libs/A.swift libs/AA.swift
	cp $(SRCDIR)/libs/B.swift libs/BB.swift
	cp $(SRCDIR)/libs/C.swift libs/CC.swift

staticA:
	# Because we override VPATH to point to the build directory, we need to pass
	# down SWIFTC too (it's default value is relative to VPATH)
	$(MAKE) MAKE_DSYM=$(MAKE_DSYM) CC=$(CC) SWIFTC=$(SWIFTC) \
		ARCH=$(ARCH) DSYMUTIL=$(DSYMUTIL) VPATH=$(BUILDDIR) SWIFTSDKROOT=$(SWIFTSDKROOT) \
		SWIFT_MODULE_CACHE_FLAGS="$(SWIFT_MODULE_CACHE_FLAGS)" \
		BASENAME=AA -f $(SRCDIR)/libs/Makefile static_only

staticB:
	$(MAKE) MAKE_DSYM=$(MAKE_DSYM) CC=$(CC) SWIFTC=$(SWIFTC) \
		ARCH=$(ARCH) DSYMUTIL=$(DSYMUTIL) VPATH=$(BUILDDIR) SWIFTSDKROOT=$(SWIFTSDKROOT) \
		SWIFT_MODULE_CACHE_FLAGS="$(SWIFT_MODULE_CACHE_FLAGS)" \
		BASENAME=BB SWIFTFLAGS_EXTRAS="-I$(BUILDDIR)" \
		LD_EXTRAS="-L$(BUILDDIR)" -f $(SRCDIR)/libs/Makefile static_only

staticC:
	$(MAKE) MAKE_DSYM=$(MAKE_DSYM) CC=$(CC) SWIFTC=$(SWIFTC) \
		ARCH=$(ARCH) DSYMUTIL=$(DSYMUTIL) VPATH=$(BUILDDIR) SWIFTSDKROOT=$(SWIFTSDKROOT) \
		SWIFT_MODULE_CACHE_FLAGS="$(SWIFT_MODULE_CACHE_FLAGS)" \
		BASENAME=CC -f $(SRCDIR)/libs/Makefile static_only

clear_modules:
	# make sure we only have .swiftinterface files for the generated modules:
	# remove the copied sources and the swiftmodules from the build directory
	rm -f *.swiftmodule
	rm -rf libs
	rm -rf MCP

clean::
	$(MAKE) VPATH=$(BUILDDIR) SWIFTC=$(SWIFTC) BASENAME=AA -f $(SRCDIR)/libs/Makefile clean
	$(MAKE) VPATH=$(BUILDDIR) SWIFTC=$(SWIFTC) BASENAME=BB -f $(SRCDIR)/libs/Makefile clean
	$(MAKE) VPATH=$(BUILDDIR) SWIFTC=$(SWIFTC) BASENAME=CC -f $(SRCDIR)/libs/Makefile clean
	rm -rf libs
	rm -rf MCP
