CC ?= clang

all: testit

LEVEL = ../../make

C_SOURCES := main.c

include $(LEVEL)/Makefile.rules


testit: testit.i386 testit.x86_64
	lipo -create -o testit testit.i386 testit.x86_64

testit.i386: testit.i386.o
	$(CC) -arch i386 -o testit.i386 testit.i386.o

testit.x86_64: testit.x86_64.o
	$(CC) -arch x86_64 -o testit.x86_64 testit.x86_64.o

# FIXME: CFLAGS_NO_ARCH isn't defined anywhere, it seems.
testit.i386.o: main.c
	$(CC) $(CFLAGS_NO_ARCH) -g -arch i386 -c -o testit.i386.o main.c

testit.x86_64.o: main.c
	$(CC) $(CFLAGS_NO_ARCH) -g -arch x86_64 -c -o testit.x86_64.o main.c

clean::
	rm -rf $(wildcard testit* *~)
