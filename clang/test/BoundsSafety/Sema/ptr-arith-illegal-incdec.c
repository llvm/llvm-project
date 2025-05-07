
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s
#include <ptrcheck.h>

int* foo(int *a) {
    ++a; // expected-error{{pointer arithmetic on single pointer 'a' is out of bounds; consider adding '__counted_by' to 'a'}}
    // expected-note@-2{{pointer 'a' declared here}}
    return a++; // expected-error{{pointer arithmetic on single pointer 'a' is out of bounds; consider adding '__counted_by' to 'a'}}
    // expected-note@-4{{pointer 'a' declared here}}
}


int main() {
    int arr[1];
    int *__indexable ap = arr;
    ap--; // expected-error{{decremented indexable pointer 'ap' is out of bounds}}
    --ap; // expected-error{{decremented indexable pointer 'ap' is out of bounds}}
    int *inc = foo(arr);
    return *inc;
}
