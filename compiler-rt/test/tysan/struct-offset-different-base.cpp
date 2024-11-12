// RUN: %clangxx_tysan -O0 %s -o %t && %run %t >%t.out 2>&1
// RUN: FileCheck %s --implicit-check-not ERROR < %t.out

// Modified reproducer from https://github.com/llvm/llvm-project/issues/105960

#include <stdio.h>

struct inner {
	char buffer;
	int i;
};

void init_inner(inner *iPtr) {
	iPtr->i = 200;
}

struct outer {
	inner foo;
    char buffer;
};

int main(void) {
    outer *l = new outer();

    init_inner(&l->foo);
    
    int access_offsets_with_different_base = l->foo.i;
    printf("Accessed value is %d\n", access_offsets_with_different_base);
    
    return 0;
}

// CHECK: Accessed value is 200
