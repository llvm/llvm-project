#include "A.h"

int main() {
    int x = 42;
    A *p = new A(&x);

    modifyPointer(p);

    A *q = p;

    useAlias(*q);

    return 0;
}