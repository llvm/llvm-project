CXX_SOURCES := main.cpp test.cpp

include Makefile.rules

a.out: dummy

dummy:
	$(MAKE) VPATH=$(VPATH) -I $(SRCDIR) -f $(SRCDIR)/dummy.mk

clean::
	$(MAKE) VPATH=$(VPATH) -I $(SRCDIR) -f $(SRCDIR)/dummy.mk clean
