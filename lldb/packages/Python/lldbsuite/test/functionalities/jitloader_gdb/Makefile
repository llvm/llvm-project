LEVEL = ../../make

C_SOURCES := main.c

all: a.out simple

include $(LEVEL)/Makefile.rules

simple:
	$(MAKE) VPATH=$(VPATH) -f $(SRCDIR)/simple.mk

clean::
	$(MAKE) -f $(SRCDIR)/simple.mk clean