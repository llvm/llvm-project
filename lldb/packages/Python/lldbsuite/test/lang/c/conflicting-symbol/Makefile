LEVEL := ../../../make

LD_EXTRAS := -L. -LOne -l$(LIB_PREFIX)One -LTwo -l$(LIB_PREFIX)Two
C_SOURCES := main.c

all: lib_One lib_Two a.out

include $(LEVEL)/Makefile.rules

lib_%:
	$(MAKE) VPATH=$(SRCDIR)/$* -I $(SRCDIR) -f $(SRCDIR)/$*.mk

clean::
	$(MAKE) -f $(SRCDIR)/One.mk clean
	$(MAKE) -f $(SRCDIR)/Two.mk clean
