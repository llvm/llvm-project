#include "A.h"

A::A(int *val) : data(val) {}

int A::getValue() const {
    return *data; //
}