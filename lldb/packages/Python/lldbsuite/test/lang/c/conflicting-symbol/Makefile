LEVEL := ../../../make

LD_EXTRAS := -L. -l$(LIB_PREFIX)One -l$(LIB_PREFIX)Two
C_SOURCES := main.c

main.o : CFLAGS_EXTRAS += -g -O0

include $(LEVEL)/Makefile.rules

.PHONY:
a.out: lib_One lib_Two

lib_%:
	$(MAKE) -f $*.mk

clean::
	$(MAKE) -f One.mk clean
	$(MAKE) -f Two.mk clean
