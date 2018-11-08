LEVEL = ../../../make

EXE=
override MAKE_DSYM=NO

include $(LEVEL)/Makefile.rules

MACOSX_DEPLOYMENT_TARGET ?= 10.10
SWIFT_TRIPLE ?= -target x86_64-apple-macosx$(MACOSX_DEPLOYMENT_TARGET)

SWIFTFLAGS += -Xfrontend -enable-resilience $(SWIFT_TRIPLE)

all: libmod.a.dylib libmod.b.dylib main.a main.b

libmod.%.dylib: mod.%.swift
	ln -sf $< mod.swift
	$(SWIFTC) $(SWIFTFLAGS) -emit-module -module-name mod mod.swift -emit-library -o $@ -Xlinker -install_name -Xlinker @executable_path/libmod.$*.dylib
ifneq "$(CODESIGN)" ""
	$(CODESIGN) -s - "$@"
endif
	mv mod.swiftdoc mod.$*.swiftdoc
	mv mod.swiftmodule mod.$*.swiftmodule
	rm mod.swift

main.%: main.swift
	ln -sf libmod.$*.dylib libmod.dylib
	ln -sf libmod.$*.dylib.dSYM libmod.dylib.dSYM
	ln -sf mod.$*.swiftdoc mod.swiftdoc
	ln -sf mod.$*.swiftmodule mod.swiftmodule
	$(SWIFTC) $(SWIFTFLAGS) $< -o main -L$(shell pwd) -I$(shell pwd) -lmod
ifneq "$(CODESIGN)" ""
	$(CODESIGN) -s - main
endif
	mv main main.$*
	mv main.dSYM main.$*.dSYM
	rm libmod.dylib libmod.dylib.dSYM mod.swiftdoc mod.swiftmodule

clean::
	rm -rf main main.a main.b *.dylib *.dSYM *.swiftdoc *.swiftmodule mod.swift

