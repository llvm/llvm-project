LEVEL := ../../../make

LD_EXTRAS := -L. -l$(LIB_PREFIX)a -l$(LIB_PREFIX)b
CXX_SOURCES := main.cpp

include $(LEVEL)/Makefile.rules

.PHONY:
a.out: lib_a lib_b

lib_%:
	$(MAKE) -f $*.mk

hidden_lib_d:
	$(MAKE) -C hidden

clean::
	$(MAKE) -f a.mk clean
	$(MAKE) -f b.mk clean
