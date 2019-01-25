LEVEL = ../../../make

EXE=main
SWIFT_SOURCES=main.swift
SWIFTFLAGS_EXTRAS = -I$(shell pwd)/moda_dir -I$(shell pwd)/modb_dir -I$(SRCDIR)/moda_dir/cdeps -I$(SRCDIR)/modb_dir/cdeps
LD_EXTRAS = -L$(shell pwd) -lmoda -lmodb

all: libmoda.dylib libmodb.dylib main

include $(LEVEL)/Makefile.rules

libmoda.dylib: moda_dir/moda.swift
libmodb.dylib: modb_dir/modb.swift

libmod%.dylib:
	echo '#############################' $<
	rm -rf mod$*_dir; mkdir mod$*_dir
	$(MAKE) -C mod$*_dir SWIFTFLAGS_EXTRAS=-I$(dir $<)/cdeps \
                MAKE_DSYM=YES CC=$(CC) SWIFTC=$(SWIFTC) \
		ARCH=$(ARCH) DSYMUTIL=$(DSYMUTIL) \
		BASENAME=$(shell basename $< .swift) \
		VPATH=$(SRCDIR)/mod$*_dir -I $(SRCDIR)/mod$*_dir \
                -f $(SRCDIR)/dylib.mk all
	ln -sf $(shell pwd)/mod$*_dir/$@ $@
	ln -sf $(shell pwd)/mod$*_dir/$@.dSYM $@.dSYM

clean::
	rm -rf a.out.dSYM a.out moda_dir/libmoda.dylib moda_dir/libmoda.dylib.dSYM modb_dir/libmodb.dylib modb_dir/libmodb.dylib.dSYM moda_dir/moda.swiftdoc moda_dir/moda.swiftmodule modb_dir/modb.swiftmodule modb_dir/modb.swiftdoc libmoda.dylib libmodb.dylib libmoda.dylib.dSYM libmodb.dylib.dSYM
