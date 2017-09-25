CC ?= clang

all: clean
	mkdir hide.app
	mkdir hide.app/Contents
	$(CC) -g main.c
	mv a.out.dSYM hide.app/Contents
	strip -x a.out

clean:
	rm -rf a.out a.out.dSYM hide.app
