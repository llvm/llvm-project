LEVEL = ../../make

CXX_SOURCES := secondprog.cpp

all: secondprog

secondprog:
	$(CXX) $(CXXFLAGS) -o secondprog $(SRCDIR)/secondprog.cpp

clean::
	rm -rf secondprog secondprog.dSYM

include $(LEVEL)/Makefile.rules
