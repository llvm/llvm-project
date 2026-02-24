// RUN: %clangxx_tysan %s -o %t && %run %t 2>&1 | FileCheck --implicit-check-not ERROR %s

#include <stdio.h>

class Inner {
public:
  void *ptr = nullptr;
};

class Base {
public:
  void *buffer1;
  Inner inside;
  void *buffer2;
};

class Derrived : public Base {};

Derrived derr;

int main() {
  printf("%p", derr.inside.ptr);

  return 0;
}
