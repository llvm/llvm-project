LEVEL = ../../../make

include $(LEVEL)/Makefile.rules

MACOSX_DEPLOYMENT_TARGET ?= 10.10
SWIFT_TRIPLE ?= -target x86_64-apple-macosx$(MACOSX_DEPLOYMENT_TARGET)
SDK_PATH ?= $(shell xcrun --show-sdk-path --sdk macosx)

everything: moda modb main

moda:
	cd moda_dir; $(SWIFTCC) $(SWIFT_TRIPLE) -sdk $(SDK_PATH) -g -Onone -Xfrontend -serialize-debugging-options -emit-module -module-name moda moda.swift -Icdeps -emit-library -o libmoda.dylib; ln -sf $(PWD)/moda_dir/libmoda.dylib $(PWD)/libmoda.dylib; ln -sf $(PWD)/moda_dir/libmoda.dylib.dSYM $(PWD)/libmoda.dylib.dSYM

modb:
	cd modb_dir; $(SWIFTCC) $(SWIFT_TRIPLE) -sdk $(SDK_PATH) -g -Onone -Xfrontend -serialize-debugging-options -emit-module -module-name modb modb.swift -Icdeps -emit-library -o libmodb.dylib; ln -sf $(PWD)/modb_dir/libmodb.dylib $(PWD)/libmodb.dylib ; ln -sf $(PWD)/modb_dir/libmodb.dylib.dSYM $(PWD)/libmodb.dylib.dSYM

main:
	$(SWIFTCC) $(SWIFT_TRIPLE) -sdk $(SDK_PATH)  -g -Onone main.swift -o a.out -Imoda_dir -Imodb_dir -Imoda_dir/cdeps -Imodb_dir/cdeps -L. -lmoda -lmodb

cleanup:
	rm -rf a.out.dSYM a.out moda_dir/libmoda.dylib moda_dir/libmoda.dylib.dSYM modb_dir/libmodb.dylib modb_dir/libmodb.dylib.dSYM moda_dir/moda.swiftdoc moda_dir/moda.swiftmodule modb_dir/modb.swiftmodule modb_dir/modb.swiftdoc libmoda.dylib libmodb.dylib libmoda.dylib.dSYM libmodb.dylib.dSYM
