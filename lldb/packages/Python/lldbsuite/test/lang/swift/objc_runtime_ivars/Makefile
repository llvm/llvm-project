LEVEL = ../../../make

SWIFT_SOURCES := main.swift

all: aTestFramework.framework a.out

include $(LEVEL)/Makefile.rules

SWIFTFLAGS += -framework aTestFramework -framework Cocoa -F $(shell pwd) -Xlinker -rpath -Xlinker $(shell pwd)

aTestFramework.framework:
	xcodebuild -configuration Release -project $(SRCDIR)/framework/aTestFramework/aTestFramework.xcodeproj CONFIGURATION_BUILD_DIR=$(shell pwd)
	cp $(SRCDIR)/module.map $(shell pwd)/aTestFramework.framework/



