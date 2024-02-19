#pragma once

class A {
  public:
    int *data;

    A(int *val) : data(val) {}

    int getValue() const {
        return *data; // sink
    }
};

void modifyPointer(A *&ptr);
int useAlias(const A &alias);
