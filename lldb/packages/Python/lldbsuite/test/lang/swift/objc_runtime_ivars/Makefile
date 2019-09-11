LEVEL = ../../../make

SWIFT_SOURCES := main.swift

all: aTestFramework.framework a.out

include $(LEVEL)/Makefile.rules

SWIFTFLAGS += -framework aTestFramework -framework Cocoa -F $(shell pwd) -Xlinker -rpath -Xlinker $(shell pwd)

CFLAGS += -arch $(ARCH)
LD_FLAGS := -dynamiclib

aTestFramework.framework:
	$(CXX) $(CFLAGS) $(LD_FLAGS) -o aTestFramework $(SRCDIR)/MyClass.mm -I $(SRCDIR) -framework Foundation
	mkdir -p aTestFramework.framework/Versions/A/Headers
	mkdir -p aTestFramework.framework/Versions/A/Modules
	mkdir -p aTestFramework.framework/Versions/A/Resources
	(cd aTestFramework.framework/Versions; ln -s A Current )
	(cd aTestFramework.framework; ln -s Versions/Current/Headers Headers)
	(cd aTestFramework.framework; ln -s Versions/Current/Modules Modules)
	(cd aTestFramework.framework; ln -s Versions/Current/Resources Resources)
	mv aTestFramework aTestFramework.framework/Versions/A
	[ -d aTestFramework.dSYM ] && mv aTestFramework.dSYM aTestFramework.framework/Versions/A
	(cd aTestFramework.framework; ln -s Versions/Current/aTestFramework aTestFramework)
	cp $(SRCDIR)/MyClass.h aTestFramework.framework/Versions/A/Headers/aTestFramework.h
	cp $(SRCDIR)/module.map $(shell pwd)/aTestFramework.framework/

clean::
	rm -rf aTestFramework.framework aTestFramework aTestFramework.dSYM
