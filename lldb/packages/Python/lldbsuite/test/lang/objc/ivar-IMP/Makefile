LEVEL = ../../../make

CC ?= clang
ifeq "$(ARCH)" ""
        ARCH = x86_64
endif

ifeq "$(OS)" ""
        OS = $(shell uname -s)
endif

CFLAGS ?= -g -O0
CFLAGS_NO_DEBUG = 
ifeq "$(OS)" "Darwin"
        CFLAGS += -arch $(ARCH)
		CFLAGS_NO_DEBUG += -arch $(ARCH)
endif

all: aout

aout: 
	$(CC) $(CFLAGS_NO_DEBUG) $(SRCDIR)/myclass.m -c -o myclass.o
	$(CC) $(CFLAGS) myclass.o $(SRCDIR)/repro.m -framework Foundation

clean::
	rm -f myclass.o

include $(LEVEL)/Makefile.rules
