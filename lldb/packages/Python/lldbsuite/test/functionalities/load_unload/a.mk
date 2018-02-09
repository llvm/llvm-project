LEVEL := ../../make

LIB_PREFIX := loadunload_

LD_EXTRAS := -L. -l$(LIB_PREFIX)b

DYLIB_NAME := $(LIB_PREFIX)a
DYLIB_CXX_SOURCES := a.cpp
DYLIB_ONLY := YES

include $(LEVEL)/Makefile.rules

$(DYLIB_FILENAME): lib_b

.PHONY lib_b:
	$(MAKE) VPATH=$(SRCDIR) -I $(SRCDIR) -f $(SRCDIR)/b.mk

clean::
	$(MAKE) -I $(SRCDIR) -f $(SRCDIR)/b.mk clean
