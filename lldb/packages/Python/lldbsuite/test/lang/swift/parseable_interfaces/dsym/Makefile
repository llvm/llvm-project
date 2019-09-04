LEVEL := ../../../../make

EXE := a.out
SWIFT_SOURCES := main.swift
LD_EXTRAS = -L$(BUILDDIR) -lAA -Xlinker -rpath -Xlinker $(BUILDDIR)
SWIFTFLAGS_EXTRAS = -I$(BUILDDIR)
SWIFT_MODULE_CACHE_FLAGS = -module-cache-path cache
DYLIB_DSYM ?= NO

# setup `all` to run each of the below before building the main executable
all: setup sharedA clear_modules

include $(LEVEL)/Makefile.rules

setup:
	# copy the source files into the build directory because we delete them
	# to be really sure we only have/use the .dylibs and .swiftinterfaces
	mkdir -p libs
	cp $(SRCDIR)/libs/A.swift libs/AA.swift

sharedA:
	$(MAKE) MAKE_DSYM=$(DYLIB_DSYM) CC=$(CC) SWIFTC=$(SWIFTC) \
		ARCH=$(ARCH) VPATH=$(BUILDDIR) SWIFTSDKROOT=$(SWIFTSDKROOT) \
		SWIFT_MODULE_CACHE_FLAGS="$(SWIFT_MODULE_CACHE_FLAGS)" \
		DSYMUTIL=$(DSYMUTIL) \
		BASENAME=AA -f $(SRCDIR)/libs/Makefile

clear_modules:
	# make sure we only have .swiftinterface files for the generated modules:
	# remove the copied sources and the swiftmodules from the build directory
	rm -f *.swiftmodule
	rm -rf libs
	rm -rf cache

clean::
	$(MAKE) VPATH=$(BUILDDIR) SWIFTC=$(SWIFTC) BASENAME=AA -f $(SRCDIR)/libs/Makefile clean
	rm -rf libs
	rm -rf cache
