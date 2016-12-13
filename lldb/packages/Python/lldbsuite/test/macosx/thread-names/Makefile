CC ?= clang
ifeq "$(ARCH)" ""
	ARCH = x86_64
endif

ifeq "$(OS)" ""
	OS = $(shell uname -s)
endif

CFLAGS ?= -g -O0
CWD := $(shell pwd)

LIB_PREFIX := lib

ifeq "$(OS)" "Darwin"
	CFLAGS += -arch $(ARCH)
endif

all: a.out

a.out: main.o
	$(CC) $(CFLAGS) -o a.out main.o

main.o: main.c
	$(CC) $(CFLAGS) -c main.c

clean:
	rm -rf $(wildcard *.o *~ *.dylib *.so a.out *.dSYM)
