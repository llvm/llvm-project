LEVEL = ../../make

C_SOURCES := main.c
LD_EXTRAS := -L. -lfoo

TRIPLE := x86_64-apple-ios13.0-macabi
CFLAGS_EXTRAS := -target $(TRIPLE)

all: libfoo.dylib a.out

lib%.dylib: %.c
	$(MAKE) MAKE_DSYM=YES CC=$(CC) \
		ARCH=$(ARCH) DSYMUTIL=$(DSYMUTIL) \
		BASENAME=$(shell basename $< .c) \
		TRIPLE=x86_64-apple-macosx10.15 SDKROOT=$(SDKROOT) \
		VPATH=$(SRCDIR) -I $(SRCDIR) -f $(SRCDIR)/dylib.mk all

include $(LEVEL)/Makefile.rules
