#include "A.h"

int main() {
    int x = 42;
    A *p = new A(&x);

    modifyPointer(p);

    A *q = p;

    useAlias(*q);

    return 0;
}

A *fooHasReturnNull() { return nullptr; }
A *fooNoReturnNull() { return new A(new int); }

void npeSources() {
    A *a1 = 0;
    A *a2 = nullptr;

    A *c1 = fooHasReturnNull();
    A *c2 = fooNoReturnNull();

    A *b1 = 0;
    A *b2 = nullptr;
}