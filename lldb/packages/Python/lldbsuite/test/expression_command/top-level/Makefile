LEVEL = ../../make

CXX_SOURCES := main.cpp test.cpp

include $(LEVEL)/Makefile.rules

a.out: dummy

dummy:
	$(MAKE) -f dummy.mk

clean::
	$(MAKE) -f dummy.mk clean
