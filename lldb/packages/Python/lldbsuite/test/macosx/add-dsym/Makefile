CC ?= clang
ifeq "$(ARCH)" ""
	ARCH = x86_64
endif

ifeq "$(OS)" ""
	OS = $(shell uname -s)
endif

CFLAGS ?= -g -O0

ifeq "$(OS)" "Darwin"
	CFLAGS += -arch $(ARCH)
endif

all: main.c clean
	mkdir hide.app
	mkdir hide.app/Contents
	$(CC) $(CFLAGS) -g $<
	mv a.out.dSYM hide.app/Contents
	strip -x a.out

clean:
	rm -rf a.out a.out.dSYM hide.app
