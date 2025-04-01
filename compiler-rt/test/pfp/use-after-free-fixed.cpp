// RUN: %clangxx_pfp %s -o %t1
// RUN: %run %t1 2>&1
// RUN: %clangxx %s -o %t2
// RUN: %run %t2 2>&1

#include <iostream>

// Struct1.ptr and Struct2.ptr have different locks.
struct Struct1 {
  int *ptr;
  Struct1() : num(1), ptr(&num) {}

private:
  int num;
};

struct Struct2 {
  int *ptr;
  Struct2() : num(2), ptr(&num) {}

private:
  int num;
};

Struct1 *new_object1() {
  Struct1 *ptr = new Struct1;
  return ptr;
}

Struct2 *new_object2() {
  Struct2 *ptr = new Struct2;
  return ptr;
}

int main() {
  Struct1 *obj1 = new_object1();
  Struct2 *obj2 = new_object2();
  std::cout << "Struct2: " << *(obj2->ptr) << "\n";
  std::cout << "Struct1: " << *(obj1->ptr) << "\n";
  delete obj1;
  delete obj2;
  return 0;
}
