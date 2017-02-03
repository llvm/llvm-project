# Build swift modules with debug info

LEVEL=../../../../make

# Don't use 'all' target.  There is a default build rule that will kick in that
# will be wrong.  WE use 'first' so that the normal 'make' command (without
# a target) selects the first (but not 'all') target so we avoid the undesired
# default behavior.
first: main

SWIFT_OBJC_INTEROP=1

include $(LEVEL)/Makefile.rules

# Add back the SDK settings to the swift flags.  Normally this happens
# automatically, but since we're overriding the normal swiftc invocation,
# we're not specifying SWIFT_SOURCES, and thus don't get the SDK.
SWIFTFLAGS+=-sdk "$(SWIFTSDKROOT)"

# To use the path commented out below, which is what we'd really want to do,
# we'd also need to require that the Swift standard library be built along
# with the compiler.  I'd like to avoid that requirement.
# SWIFT_LIB_DIR=$(dir $(SWIFTCC))../lib
SWIFT_LIB_DIR="$(shell xcode-select -p)/Toolchains/XcodeDefault.xctoolchain/usr/lib/swift/macosx"

main: A.o B.o objc_main.m
	$(CC) $(CFLAGS) objc_main.m -fobjc-arc -o main A.o B.o -L $(SWIFT_LIB_DIR) -Xlinker -add_ast_path -Xlinker A.swiftmodule -Xlinker -add_ast_path -Xlinker B.swiftmodule -Xlinker -rpath -Xlinker $(SWIFT_LIB_DIR)

A.o: A.swift
	$(SWIFTCC) $(SWIFTFLAGS) -c -parse-as-library -module-name A -emit-module-path A.swiftmodule -emit-objc-header-path A-Swift.h -output-file-map output_map A.swift

B.o: B.swift
	$(SWIFTCC) $(SWIFTFLAGS) -c -parse-as-library -module-name B -emit-module-path B.swiftmodule -emit-objc-header-path B-Swift.h -output-file-map output_map B.swift -o B.o

clean::
	rm -f *.o main *-Swift.h *.swiftmodule *.swiftdoc
