// RUN: %clangxx_cfi %s -o %t
// RUN: %run %t

#include <memory>

struct Foo {
  Foo() {}
  virtual ~Foo() {}
};

int main(int argc, char **argv) { std::make_shared<Foo>(); }
