LEVEL := ../../make

EXE := testit

include $(LEVEL)/Makefile.rules

all: testit

testit: testit.x86_64h testit.x86_64
	lipo -create -o testit $^

testit.x86_64h: testit.x86_64h.o
	$(CC) -arch x86_64h -o testit.x86_64h $<

testit.x86_64: testit.x86_64.o
	$(CC) -arch x86_64 -o testit.x86_64 $<

testit.x86_64h.o: main.c
	$(CC) -g -O0 -arch x86_64h -c -o testit.x86_64h.o $<

testit.x86_64.o: main.c
	$(CC) -g -O0 -arch x86_64 -c -o testit.x86_64.o $<

clean::
	rm -rf $(wildcard testit* *~)
