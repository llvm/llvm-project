LEVEL = ../../../make

CXX_SOURCES := main.cpp

EXE := AttachDenied

all: AttachDenied sign

include $(LEVEL)/Makefile.rules

sign: entitlements.plist AttachDenied
ifeq ($(OS),Darwin)
	codesign -s - -f --entitlements $^
endif
