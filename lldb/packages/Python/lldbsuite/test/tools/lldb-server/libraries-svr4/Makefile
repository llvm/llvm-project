LEVEL = ../../../make

LIB_PREFIX := svr4lib
LD_EXTRAS := -L. -l$(LIB_PREFIX)_a -l$(LIB_PREFIX)_b\"
CXX_SOURCES := main.cpp
USE_LIBDL := 1
MAKE_DSYM := NO

include $(LEVEL)/Makefile.rules

a.out: $(LIB_PREFIX)_a $(LIB_PREFIX)_b_quote

svr4lib_%:
	$(MAKE) VPATH=$(SRCDIR) -I $(SRCDIR) -f "$(SRCDIR)/$(LIB_PREFIX)_$*.mk"

clean::
	$(MAKE) -f $(SRCDIR)/$(LIB_PREFIX)_a.mk clean
