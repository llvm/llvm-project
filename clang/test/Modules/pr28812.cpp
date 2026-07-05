// RUN: rm -rf %t
// RUN: %clang_cc1 -std=c++11 -nostdsysteminc -I%S/Inputs/PR28812 -verify %s
// RUN: %clang_cc1 -std=c++11 -nostdsysteminc -fmodules -fimplicit-module-maps \
// RUN:            -fmodules-cache-path=%t -I%S/Inputs/PR28812 -verify %s

template <typename> struct VarStreamArrayIterator;
template <typename ValueType>
struct VarStreamArray {
  typedef VarStreamArrayIterator<ValueType> Iterator;
  Iterator begin() { return Iterator(*this); }
};

#include "Textual.h"

#include "a.h"
#include "b.h"

VarStreamArray<int> a;
auto b = a.begin();

// expected-no-diagnostics

