LEVEL := ../../../make

DYLIB_NAME := One
DYLIB_C_SOURCES := One.c OneConstant.c
DYLIB_ONLY := YES

include $(LEVEL)/Makefile.rules

CFLAGS_EXTRAS += -fPIC

OneConstant.o: OneConstant.c
	$(CC) $(CFLAGS_NO_DEBUG) -c $< -o $@
