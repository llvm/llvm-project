LEVEL = ../../../make
SWIFT_SOURCES := main.swift

all: libInvisible.dylib libLibrary.dylib a.out

include $(LEVEL)/Makefile.rules
LD_EXTRAS = -lLibrary -L$(BUILDDIR)
SWIFTFLAGS_EXTRAS = -I$(BUILDDIR)

libInvisible.dylib: Invisible.swift
	$(MAKE) MAKE_DSYM=NO CC=$(CC) SWIFTC=$(SWIFTC) \
		ARCH=$(ARCH) DSYMUTIL=$(DSYMUTIL) \
		BASENAME=Invisible DEBUG_INFO_FLAG=-gnone \
		VPATH=$(SRCDIR) -I $(SRCDIR) -f $(SRCDIR)/dylib.mk all

libLibrary.dylib: Library.swift
	$(MAKE) MAKE_DSYM=YES CC=$(CC) SWIFTC=$(SWIFTC) \
		ARCH=$(ARCH) DSYMUTIL=$(DSYMUTIL) \
		BASENAME=Library \
		LD_EXTRAS="-lInvisible -L$(BUILDDIR)" \
		SWIFTFLAGS_EXTRAS="-I$(BUILDDIR) -enable-library-evolution" \
		VPATH=$(SRCDIR) -I $(SRCDIR) -f $(SRCDIR)/dylib.mk all

clean::
	$(MAKE) BASENAME=Invisible VPATH=$(SRCDIR) -I $(SRCDIR) -f $(SRCDIR)/dylib.mk clean
	$(MAKE) BASENAME=Library VPATH=$(SRCDIR) -I $(SRCDIR) -f $(SRCDIR)/dylib.mk clean
