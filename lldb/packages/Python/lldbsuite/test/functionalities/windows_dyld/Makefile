LEVEL := ../../make

LD_EXTRAS := -ldllfunc
C_SOURCES := main.c

include $(LEVEL)/Makefile.rules

a.out: dllfunc

dllfunc:
	$(MAKE) VERBOSE=1 VPATH=$(SRCDIR) -I $(SRCDIR) -f $(SRCDIR)/dllfunc.mk

clean::
	$(MAKE) -f $(SRCDIR)/dllfunc.mk clean
