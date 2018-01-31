LEVEL := ../../../make

DYLIB_NAME := Two
DYLIB_C_SOURCES := Two.c TwoConstant.c
DYLIB_ONLY := YES

include $(LEVEL)/Makefile.rules

CFLAGS_EXTRAS += -fPIC

TwoConstant.o: TwoConstant.c
	$(CC) $(CFLAGS_NO_DEBUG) -c $< -o $@
