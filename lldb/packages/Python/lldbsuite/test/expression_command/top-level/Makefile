LEVEL = ../../make

CXX_SOURCES := main.cpp test.cpp

all: dummy a.out

include $(LEVEL)/Makefile.rules

dummy:
	$(MAKE) VPATH=$(VPATH) -I $(SRCDIR) -f $(SRCDIR)/dummy.mk

clean::
	$(MAKE) VPATH=$(VPATH) -I $(SRCDIR) -f $(SRCDIR)/dummy.mk clean
