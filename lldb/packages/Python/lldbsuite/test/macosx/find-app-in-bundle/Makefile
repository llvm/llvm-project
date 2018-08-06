SRCDIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))/

CC ?= clang

ifeq "$(ARCH)" ""
     ARCH = x86_64
endif

CFLAGS ?= -g -O0 -arch $(ARCH)

all:  TestApp.app/Contents/MacOS/TestApp

TestApp.app/Contents/MacOS/TestApp: $(SRCDIR)/main.c
	$(CC) $(CFLAGS) -o TestApp $<
	rm -rf TestApp.app
	cp -r $(SRCDIR)/TestApp.app .
	mv TestApp TestApp.app/Contents/MacOS/TestApp
	mv TestApp.dSYM TestApp.app.dSYM

clean:
	rm -rf TestApp.app/Contents/MacOS/TestApp TestApp.app.dSYM
