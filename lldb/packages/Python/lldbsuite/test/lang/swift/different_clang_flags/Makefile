LEVEL = ../../../make

all: libmoda.dylib libmodb.dylib main

EXE=main
SWIFT_SOURCES=main.swift
LINKFLAGS=-I$(shell pwd)/moda_dir -I$(shell pwd)/modb_dir -I$(SRCDIR)/moda_dir/cdeps -I$(SRCDIR)/modb_dir/cdeps -L$(shell pwd) -lmoda -lmodb
SWIFTFLAGS=-g -Onone $(LINKFLAGS) -Xfrontend -serialize-debugging-options

include $(LEVEL)/Makefile.rules

SWIFTFLAGS_NOLINK = $(subst $(LINKFLAGS),,$(SWIFTFLAGS))

libmoda.dylib: moda_dir/moda.swift
libmodb.dylib: modb_dir/modb.swift

libmod%.dylib:
	rm -rf mod$*_dir; mkdir mod$*_dir
	cd mod$*_dir; $(SWIFTC) $(SWIFTFLAGS_NOLINK) -emit-module -module-name mod$* $< -I$(dir $<)/cdeps -emit-library -o $@ -Xlinker -install_name -Xlinker @executable_path/$@
ifneq "$(CODESIGN)" ""
	$(CODESIGN) -s - "$@"
endif
	ln -sf $(shell pwd)/mod$*_dir/$@ $@
	ln -sf $(shell pwd)/mod$*_dir/$@.dSYM $@.dSYM

clean::
	rm -rf a.out.dSYM a.out moda_dir/libmoda.dylib moda_dir/libmoda.dylib.dSYM modb_dir/libmodb.dylib modb_dir/libmodb.dylib.dSYM moda_dir/moda.swiftdoc moda_dir/moda.swiftmodule modb_dir/modb.swiftmodule modb_dir/modb.swiftdoc libmoda.dylib libmodb.dylib libmoda.dylib.dSYM libmodb.dylib.dSYM
