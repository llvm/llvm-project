CXX_SOURCES := main.cpp other.cpp other-2.cpp
C_SOURCES := somefunc.c

include Makefile.rules

other-2.o: other-2.cpp
	$(CXX) $(CFLAGS_NO_DEBUG) -c $(SRCDIR)/other-2.cpp

somefunc.o: somefunc.c
	$(CC) $(CFLAGS) -std=c99 -c $(SRCDIR)/somefunc.c
