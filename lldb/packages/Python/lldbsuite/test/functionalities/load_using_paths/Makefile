LEVEL := ../../make

CXX_SOURCES := main.cpp
USE_LIBDL := 1

include $(LEVEL)/Makefile.rules

all: hidden_lib a.out

hidden_lib:
	$(MAKE) VPATH=$(SRCDIR)/hidden -I $(SRCDIR)/hidden -C hidden -f $(SRCDIR)/hidden/Makefile

clean::
	$(MAKE) -I $(SRCDIR)/hidden -C hidden -f $(SRCDIR)/hidden/Makefile clean
