LEVEL := ../../../make

DYLIB_NAME := One
DYLIB_C_SOURCES := One/One.c One/OneConstant.c
DYLIB_ONLY := YES

include $(LEVEL)/Makefile.rules

CFLAGS_EXTRAS += -fPIC

One/OneConstant.o: One/OneConstant.c
	$(CC) $(CFLAGS_NO_DEBUG) -c $< -o $@
