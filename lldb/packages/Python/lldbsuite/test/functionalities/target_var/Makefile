LEVEL = ../../make

include $(LEVEL)/Makefile.rules

a.out: globals.ll
	$(CC) $(CFLAGS) -g -c $^ -o globals.o
	$(LD) $(LDFLAGS) -g globals.o -o $@

clean::
	rm -rf globals.o a.out *.dSYM
