LEVEL = ../../../make

myclass.o: myclass.h myclass.m
	$(CC) myclass.m -c -o myclass.o

repro: myclass.o repro.m
	$(CC) -g -O0 myclass.o repro.m -framework Foundation

cleanup:
	rm -r myclass.o

include $(LEVEL)/Makefile.rules
