LEVEL := ../../make

LIB_PREFIX := loadunload_

LD_EXTRAS := -L. -l$(LIB_PREFIX)d
CXX_SOURCES := main.cpp
USE_LIBDL := 1

all: lib_a lib_b lib_c lib_d hidden_lib_d install_name_tool

include $(LEVEL)/Makefile.rules

lib_%:
	$(MAKE) VPATH=$(SRCDIR) -I $(SRCDIR) -f $(SRCDIR)/$*.mk

install_name_tool:
ifeq ($(OS),Darwin)
	install_name_tool -id @executable_path/libloadunload_d.dylib libloadunload_d.dylib
endif


hidden_lib_d:
	$(MAKE) VPATH=$(SRCDIR)/hidden -I $(SRCDIR)/hidden -C hidden -f $(SRCDIR)/hidden/Makefile

clean::
	$(MAKE) -f $(SRCDIR)/a.mk clean
	$(MAKE) -f $(SRCDIR)/b.mk clean
	$(MAKE) -f $(SRCDIR)/c.mk clean
	$(MAKE) -f $(SRCDIR)/d.mk clean
	$(MAKE) -I $(SRCDIR)/hidden -C hidden -f $(SRCDIR)/hidden/Makefile clean
