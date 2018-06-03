LEVEL = ../../../make
OBJCXX_SOURCES := main.mm myobject.mm
include $(LEVEL)/Makefile.rules

# myobject.o needs to be built without debug info
myobject.o: myobject.mm
	$(CXX) -c -o $@ $<
