LEVEL := ../../make

LIB_PREFIX := loadunload_

LD_EXTRAS := -L. -l$(LIB_PREFIX)d -ldl
CXX_SOURCES := main.cpp

include $(LEVEL)/Makefile.rules

.PHONY:
a.out: lib_a lib_b lib_c lib_d hidden_lib_d

lib_%:
	$(MAKE) -f $*.mk

hidden_lib_d:
	$(MAKE) -C hidden

clean::
	$(MAKE) -f a.mk clean
	$(MAKE) -f b.mk clean
	$(MAKE) -f c.mk clean
	$(MAKE) -f d.mk clean
	$(MAKE) -C hidden clean
