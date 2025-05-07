
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s

void *my_calloc(int count, int size) __attribute__((alloc_size(1,2)));
int get_len();

int foo() {
    int a = 10;
    int b = sizeof(int);
    int *ptr = my_calloc(get_len(), b);
    ptr = my_calloc(a, get_len());
    return ptr[10];
}

// expected-no-diagnostics
