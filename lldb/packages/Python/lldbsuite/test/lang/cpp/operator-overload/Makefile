LEVEL = ../../../make

CXX_SOURCES = a.cpp b.cpp
CXXFLAGS_NO_DEBUGINFO = -c
CXXFLAGS_DEBUGINFO = -c -g

all: main

main: a.o b.o
	$(CXX) a.o b.o -o main $(LDFLAGS)

a.o: a.cpp
	$(CXX) $(SRCDIR)/a.cpp $(CXXFLAGS_NO_DEBUGINFO) -o a.o

b.o: b.cpp
	$(CXX) $(SRCDIR)/b.cpp $(CXXFLAGS_DEBUGINFO) -o b.o

clean: OBJECTS += a.o b.o main

include $(LEVEL)/Makefile.rules
