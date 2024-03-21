#include "A.h"
#include <cstddef>

int a;
int *getPtr() {
    if (a > 0) {
        return &a;
    }
    return NULL;
}

void modifyPointer(A *&ptr) {
    ptr->data = getPtr(); // source
    return;
}