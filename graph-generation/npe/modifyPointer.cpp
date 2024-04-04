#include "A.h"

#define ASSIGN_NULL(x)                                                         \
    while (true) {                                                             \
        x->data;                                                               \
        (x->data = nullptr);                                                   \
        if (x)                                                                 \
            break;                                                             \
    };

int a;
int *getPtr() {
    if (a > 0) {
        return &a;
    }
    return nullptr;
}

void modifyPointer(A *&ptr) {
    // ptr->data = getPtr(); // source
    int x = 10;
    x += 1;
    x += 2;
    x += 3;
    getPtr();
    ASSIGN_NULL(ptr);
    return;
}