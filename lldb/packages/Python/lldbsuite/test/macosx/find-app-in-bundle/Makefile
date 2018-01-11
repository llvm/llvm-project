CC ?= clang

ifeq "$(ARCH)" ""
	ARCH = x86_64
endif

CFLAGS ?= -g -O0 -arch $(ARCH)

all: clean TestApp

TestApp:
	$(CC) $(CFLAGS) -o TestApp main.c
	mv TestApp TestApp.app/Contents/MacOs/TestApp
	mv TestApp.dSYM TestApp.app.dSYM
clean:
	rm -rf TestApp.app/Contents/MacOS/TestApp TestApp.app.dSYM
