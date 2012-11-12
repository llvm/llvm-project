GXX := llvm-g++-4.2
CLANGXX := clang++

all: one

test.cc: gen.cc
	g++ gen.cc -o gen
	./gen >test.cc

test-gcc.sum: test.cc
	time $(GXX) test.cc -o test-gcc.s -S
	$(GXX) test-gcc.s -o test-gcc
	./test-gcc >test-gcc.sum

test-clang.sum: test.cc
	time $(CLANGXX) test.cc -o test-clang.s -S
	$(CLANGXX) test-clang.s -o test-clang
	./test-clang >test-clang.sum

one: test-gcc.sum test-clang.sum
	cmp test-gcc.sum test-clang.sum

clean:
	rm -f gen test-gcc test-clang test.cc test-gcc.sum test-clang.sum test-gcc.s test-clang.s
