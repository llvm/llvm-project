#include "A.h"

int a;
int *getPtr() {
    if (a > 0) {
        return &a;
    }
    return nullptr;
}

void modifyPointer(A *&ptr) {
    ptr->data = getPtr(); // source
    return;
}