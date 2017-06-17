CC ?= clang

ifeq "$(ARCH)" ""
	ARCH = x86_64
endif

CFLAGS ?= -g -O0 -arch $(ARCH)

all: clean
	$(CC) $(CFLAGS) -dynamiclib -o com.apple.sbd bundle.c
	mkdir com.apple.sbd.xpc
	mv com.apple.sbd com.apple.sbd.xpc/
	mkdir -p com.apple.sbd.xpc.dSYM/Contents/Resources/DWARF
	mv com.apple.sbd.dSYM/Contents/Resources/DWARF/com.apple.sbd com.apple.sbd.xpc.dSYM/Contents/Resources/DWARF/
	rm -rf com.apple.sbd.dSYM
	mkdir hide.app
	tar cf - com.apple.sbd.xpc com.apple.sbd.xpc.dSYM | ( cd hide.app;tar xBpf -)
	$(CC) $(CFLAGS) -o find-bundle-with-dots-in-fn main.c

clean:
	rm -rf a.out a.out.dSYM hide.app com.apple.sbd com.apple.sbd.dSYM com.apple.sbd.xpc com.apple.sbd.xpc.dSYM find-bundle-with-dots-in-fn find-bundle-with-dots-in-fn.dSYM
