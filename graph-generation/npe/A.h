#pragma once

class A {
  public:
    int *data;

    A(int *val);

    int getValue() const;
};

void modifyPointer(A *&ptr);
int useAlias(const A &alias);
