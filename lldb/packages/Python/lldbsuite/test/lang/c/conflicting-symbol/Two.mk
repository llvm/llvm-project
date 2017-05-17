LEVEL := ../../../make

DYLIB_NAME := Two
DYLIB_C_SOURCES := Two/Two.c Two/TwoConstant.c
DYLIB_ONLY := YES

include $(LEVEL)/Makefile.rules

CFLAGS_EXTRAS += -fPIC

Two/TwoConstant.o: Two/TwoConstant.c
	$(CC) $(CFLAGS_NO_DEBUG) -c $< -o $@
