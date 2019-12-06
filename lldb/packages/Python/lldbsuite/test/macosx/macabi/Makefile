C_SOURCES := main.c
LD_EXTRAS := -L. -lfoo

TRIPLE := x86_64-apple-ios13.0-macabi
CFLAGS_EXTRAS := -target $(TRIPLE)

all: libfoo.dylib a.out

libfoo.dylib: foo.c
	$(MAKE) -f $(MAKEFILE_RULES) \
		DYLIB_ONLY=YES DYLIB_NAME=foo DYLIB_C_SOURCES=foo.c

include Makefile.rules
