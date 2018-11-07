LEVEL = ../../../make

SWIFT_SOURCES := main.swift

include $(LEVEL)/Makefile.rules

fakesdk:
	ln -s $(SDKROOT) $@

SWIFTFLAGS := $(subst $(SDKROOT),$(shell pwd)/fakesdk,$(SWIFTFLAGS))

a.out: $(SWIFT_SOURCES) fakesdk
	$(SWIFTC) $< $(SWIFTFLAGS) -o "$(EXE)"
ifneq "$(CODESIGN)" ""
	$(CODESIGN) -s - "$@"
endif
	rm -f fakesdk

clean::
	rm -f fakesdk a.out *.dSYM
