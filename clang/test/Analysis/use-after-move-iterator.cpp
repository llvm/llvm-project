// RUN: %clang_analyze_cc1 -std=c++17 \
// RUN: -analyzer-checker=core,cplusplus.Move,alpha.cplusplus.IteratorModeling \
// RUN: -analyzer-config aggressive-binary-operation-simplification=true \
// RUN: -analyzer-config c++-container-inlining=false %s \
// RUN: -verify -analyzer-config display-checker-name=false

#include "Inputs/system-header-simulator-cxx.h"

//===----------------------------------------------------------------------===//
// Test suite for test functions that require both MoveChecker.cpp and
// IteratorModeling.cpp to be enabled.
// NOTE: Currently the iterator dereference detection is only working when
// IteratorModeling is enabled.
//===----------------------------------------------------------------------===//

enum Target {MovedFromSource, Destination};

std::string gh137157_iteratorDerefList(Target trg) {
  std::list<std::string> l1;
  l1.push_back("l1");
  std::list<std::string> l2;

  switch (trg) {
    case MovedFromSource: {
      std::move(l1.begin(), l1.end(), std::back_inserter(l2));
      return *l1.cbegin(); // expected-warning {{Method called on moved-from object 'l1'}}
    }
    case Destination: {
      std::move(l1.begin(), l1.end(), std::back_inserter(l2));
      return *l2.cbegin(); // no-warning: only l1 was invalidated and not l2!
    }
  }
  return {};
}

std::string gh137157_iteratorDerefVector(Target trg) {
  std::vector<std::string> l1;
  l1.push_back("l1");
  std::vector<std::string> l2;

  switch (trg) {
    case MovedFromSource: {
      std::move(l1.begin(), l1.end(), std::back_inserter(l2));
      return *l1.cbegin(); // expected-warning {{Method called on moved-from object 'l1'}}
    }
    case Destination: {
      std::move(l1.begin(), l1.end(), std::back_inserter(l2));
      return *l2.cbegin(); // no-warning: only l1 was invalidated and not l2!
    }
  }
  return {};
}