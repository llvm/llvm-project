LEVEL = ../../../make

MAKE_DSYM := NO

C_SOURCES := main.c

all: a.out create-empty-corefile

create-empty-corefile:
	$(MAKE) VPATH=$(VPATH) -f $(SRCDIR)/create-empty-corefile.mk

clean::
	$(MAKE) -f create-empty-corefile.mk clean

include $(LEVEL)/Makefile.rules
