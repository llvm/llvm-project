LEVEL = ../../../make

OBJC_SOURCES := main.m

CFLAGS_EXTRAS := -w

LD_EXTRAS := -framework Foundation

include $(LEVEL)/Makefile.rules

clean::
	rm -rf $(wildcard *.o *.d *.dSYM *.log)
