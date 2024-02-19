#include "A.h"

void modifyPointer(A *&ptr) {
    ptr->data = nullptr; // source
    return;
}