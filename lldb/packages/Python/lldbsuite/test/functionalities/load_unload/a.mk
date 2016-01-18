LEVEL := ../../make

LIB_PREFIX := loadunload_

CFLAGS_EXTRAS := -fPIC
LD_EXTRAS := -L. -l$(LIB_PREFIX)b

DYLIB_NAME := $(LIB_PREFIX)a
DYLIB_C_SOURCES := a.c
DYLIB_ONLY := YES

include $(LEVEL)/Makefile.rules

.PHONY:
$(DYLIB_FILENAME): lib_b

lib_b:
	$(MAKE) -f b.mk

clean::
	$(MAKE) -f b.mk clean
