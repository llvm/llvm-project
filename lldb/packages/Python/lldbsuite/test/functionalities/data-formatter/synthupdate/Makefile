OBJC_SOURCES := main.m

CFLAGS_EXTRAS += -w

include Makefile.rules

LDFLAGS += -framework Foundation

clean::
	rm -rf $(wildcard *.o *.d *.dSYM *.log)
