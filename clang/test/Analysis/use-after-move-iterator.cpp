// RUN: %clang_analyze_cc1 -std=c++17 -analyzer-checker=core,cplusplus.Move,alpha.cplusplus.IteratorModeling -analyzer-config aggressive-binary-operation-simplification=true -analyzer-config c++-container-inlining=false %s -verify -analyzer-config display-checker-name=false

#include "Inputs/system-header-simulator-cxx.h"


//===----------------------------------------------------------------------===//
// Test suite for test functions that require both MoveChecker.cpp and
// IteratorModeling.cpp to be enabled.
// NOTE: Currently the iterator dereference detection is only working when
// IteratorModeling is enabled.
//===----------------------------------------------------------------------===//

std::string iteratorDeref(int rng) {
  std::list<std::string> l1;
  l1.push_back("l1");
  std::list<std::string> l2;

  switch (rng) {
    case 10: {
      std::move(l1.begin(), l1.end(), std::back_inserter(l2));
      return *l1.cbegin(); // expected-warning {{Method called on moved-from object 'l1'}}
    }
    case 20: {
      std::move(l1.begin(), l1.end(), std::back_inserter(l2));
      return *l2.cbegin(); // no-warning: only l1 was invalidated and not l2!
    }
  }
  return 0;
}
