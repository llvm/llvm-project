LEVEL = ../../../make

OBJC_SOURCES := static.m
LDFLAGS = $(CFLAGS) -lobjc -framework Foundation

default:        a.out.stripped

a.out.stripped: a.out.dSYM
	strip -o a.out.stripped a.out
	ln -sf a.out.dSYM a.out.stripped.dSYM

clean::
	rm -f a.out.stripped
	rm -rf $(wildcard *.dSYM)

include $(LEVEL)/Makefile.rules
