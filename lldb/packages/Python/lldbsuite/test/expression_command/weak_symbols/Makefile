LEVEL = ../../make
CFLAGS_EXTRAS += -std=c99
LD_FLAGS := -dynamiclib
include $(LEVEL)/Makefile.rules

all: a.out dylib missing

dylib: dylib.o
	$(CC)  $(LD_FLAGS) -o libdylib.dylib dylib.o

missing: dylib2.o
	mkdir hidden
	$(CC)  $(LD_FLAGS) -o hidden/libdylib.dylib dylib2.o

a.out: main.o dylib missing
	$(CC)  $(CFLAGS) -L. -ldylib main.o

dylib.o: dylib.h $(SRCDIR)/dylib.c
	$(CC) -DHAS_THEM  $(CFLAGS) -c $(SRCDIR)/dylib.c

dylib2.o: dylib.h $(SRCDIR)/dylib.c
	$(CC)  $(CFLAGS) -c $(SRCDIR)/dylib.c -o dylib2.o

main.o: dylib.h $(SRCDIR)/main.c
	$(CC)  $(CFLAGS) -c $(SRCDIR)/main.c -fmodules -fmodules-cache-path=$(CLANG_MODULE_CACHE_DIR)

