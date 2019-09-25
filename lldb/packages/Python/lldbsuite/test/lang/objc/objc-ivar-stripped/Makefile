OBJC_SOURCES := main.m
LD_EXTRAS := -lobjc -framework Foundation

all:        a.out.stripped

a.out.stripped: a.out.dSYM
	strip -o a.out.stripped a.out

clean::
	rm -f a.out.stripped
	rm -rf a.out.stripped.dSYM

include Makefile.rules
