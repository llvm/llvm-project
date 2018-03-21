LEVEL = ../../../make

CXX_SOURCES = a.cpp b.cpp
CXXFLAGS_NO_DEBUGINFO = -c
CXXFLAGS_DEBUGINFO = -c -g

all: main

main: a.o b.o
	$(CXX) $^ -o $@ $(LDFLAGS)

a.o: a.cpp
	$(CXX) $< $(CXXFLAGS_NO_DEBUGINFO) -o $@

b.o: b.cpp
	$(CXX) $< $(CXXFLAGS_DEBUGINFO) -o $@

include $(LEVEL)/Makefile.rules
