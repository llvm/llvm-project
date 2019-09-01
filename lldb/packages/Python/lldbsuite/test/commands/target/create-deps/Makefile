LEVEL := ../../make

LIB_PREFIX := load_

LD_EXTRAS := -L. -l$(LIB_PREFIX)a
CXX_SOURCES := main.cpp

include $(LEVEL)/Makefile.rules

a.out: lib_a

lib_%:
	$(MAKE) VPATH=$(SRCDIR) -I $(SRCDIR) -f $(SRCDIR)/$*.mk

clean::
	$(MAKE) -f $(SRCDIR)/a.mk clean
