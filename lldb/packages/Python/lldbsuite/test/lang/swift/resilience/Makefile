LEVEL = ../../../make

include $(LEVEL)/Makefile.rules

MACOSX_DEPLOYMENT_TARGET ?= 10.10
SWIFT_TRIPLE ?= -target x86_64-apple-macosx$(MACOSX_DEPLOYMENT_TARGET)
SDK_PATH ?= $(shell xcrun --show-sdk-path --sdk macosx)

everything: libmod.a.dylib libmod.b.dylib main.a main.b

RESILIENCE_FLAGS = -Xfrontend -enable-resilience

libmod.a.dylib:
	ln -sf mod.a.swift mod.swift
	$(SWIFTCC) $(SWIFT_TRIPLE) -sdk $(SDK_PATH) -g -Onone -Xfrontend -serialize-debugging-options $(RESILEINCE_FLAGS) -emit-module -module-name mod mod.swift -emit-library -o libmod.dylib
	mv libmod.dylib libmod.a.dylib
	mv libmod.dylib.dSYM libmod.a.dylib.dSYM
	mv mod.swiftdoc mod.a.swiftdoc
	mv mod.swiftmodule mod.a.swiftmodule
	rm mod.swift

libmod.b.dylib:
	ln -sf mod.b.swift mod.swift
	$(SWIFTCC) $(SWIFT_TRIPLE) -sdk $(SDK_PATH) -g -Onone -Xfrontend -serialize-debugging-options $(RESILEINCE_FLAGS) -emit-module -module-name mod mod.swift -emit-library -o libmod.dylib
	mv libmod.dylib libmod.b.dylib
	mv libmod.dylib.dSYM libmod.b.dylib.dSYM
	mv mod.swiftdoc mod.b.swiftdoc
	mv mod.swiftmodule mod.b.swiftmodule
	rm mod.swift

main.a:
	ln -sf libmod.a.dylib libmod.dylib
	ln -sf libmod.a.dylib.dSYM libmod.dylib.dSYM
	ln -sf mod.a.swiftdoc mod.swiftdoc
	ln -sf mod.a.swiftmodule mod.swiftmodule
	$(SWIFTCC) $(SWIFT_TRIPLE) -sdk $(SDK_PATH)  -g -Onone $(RESILIENCE_FLAGS) main.swift -o main -L. -I. -lmod
	mv main main.a
	mv main.dSYM main.a.dSYM
	rm libmod.dylib libmod.dylib.dSYM mod.swiftdoc mod.swiftmodule

main.b:
	ln -sf libmod.b.dylib libmod.dylib
	ln -sf libmod.b.dylib.dSYM libmod.dylib.dSYM
	ln -sf mod.b.swiftdoc mod.swiftdoc
	ln -sf mod.b.swiftmodule mod.swiftmodule
	$(SWIFTCC) $(SWIFT_TRIPLE) -sdk $(SDK_PATH)  -g -Onone $(RESILIENCE_FLAGS) main.swift -o main -L. -I. -lmod
	mv main main.b
	mv main.dSYM main.b.dSYM
	rm libmod.dylib libmod.dylib.dSYM mod.swiftdoc mod.swiftmodule

cleanup:
	rm -rf main main.a main.b *.dylib *.dSYM *.swiftdoc *.swiftmodule mod.swift

