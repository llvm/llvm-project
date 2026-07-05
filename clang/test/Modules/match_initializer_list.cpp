// RUN: rm -rf %t
//
// RUN: %clang_cc1 -x c++ -fmodules -fmodules-cache-path=%t \
// RUN:     -I %S/Inputs/initializer_list \
// RUN:     -fmodule-map-file=%S/Inputs/initializer_list/direct.modulemap \
// RUN:     %s -verify

// expected-no-diagnostics

class C {
  public:
  virtual ~C() {}
};

#include "Inputs/initializer_list/direct.h"

void takesInitList(std::initializer_list<int>);

void passesInitList() { takesInitList({0}); }
