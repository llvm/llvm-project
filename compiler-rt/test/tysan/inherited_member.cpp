// RUN: %clangxx_tysan %s -o %t && %run %t 2>&1 | FileCheck --implicit-check-not ERROR %s

#include <stdio.h>

class Base {
public:
  void *first;
  void *second;
  void *third;
};

class Derrived : public Base {};

Derrived derr;

int main() {
  derr.second = nullptr;
  printf("%p", derr.second);

  return 0;
}
