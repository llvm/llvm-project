LEVEL = ../../../make

CFLAGS = -g -O0
LDFLAGS = $(CFLAGS) -lobjc -framework Foundation

all: a.out libTest.dylib libTestExt.dylib

libTest.dylib:	Test/Test.m
	$(CC) $(CFLAGS) -I. -c -o Test.o Test/Test.m
	$(CC) $(LDFLAGS) -shared -o libTest.dylib Test.o
	dsymutil libTest.dylib

libTestExt.dylib: TestExt/TestExt.m
	$(CC) $(CFLAGS) -I. -c -o TestExt.o TestExt/TestExt.m
	$(CC) $(LDFLAGS) -L. -lTest -shared -o libTestExt.dylib TestExt.o
	dsymutil libTestExt.dylib

a.out: main.m libTest.dylib libTestExt.dylib
	$(CC) $(LDFLAGS) -I. -L. -lTest -lTestExt -o a.out main.m 

.PHONY: clean

clean:
	rm -rf *.dylib a.out *.o *.dSYM *.d
