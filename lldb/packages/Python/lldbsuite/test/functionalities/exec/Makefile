LEVEL = ../../make

CXX_SOURCES := main.cpp

all: a.out secondprog

include $(LEVEL)/Makefile.rules

secondprog:
	$(MAKE) VPATH=$(VPATH) -f $(SRCDIR)/secondprog.mk

clean::
	$(MAKE) -f $(SRCDIR)/secondprog.mk clean
