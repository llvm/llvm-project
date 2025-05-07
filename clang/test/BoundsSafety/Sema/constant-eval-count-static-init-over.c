
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s

#include <ptrcheck.h>

typedef struct {
    unsigned int        *__counted_by(length) data;
    int        length;
} Item;

// expected-note@+1{{'_oidRsa' declared here}}
static unsigned int _oidRsa[] = { 0, 1, 2 };
// expected-error@+1{{initializing 'oidRsa.data' of type 'unsigned int *__single __counted_by(length)' (aka 'unsigned int *__single') and count value of 4 with array '_oidRsa' (which has 3 elements) always fails}}
const Item oidRsa = { _oidRsa, sizeof(_oidRsa)/sizeof(int) + 1 };

int main() {
	return oidRsa.data[3];
}
