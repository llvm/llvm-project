// RUN: %clang_analyze_cc1 -std=c++17 -verify %s \
// RUN:   -analyzer-checker=core,cplusplus.Move,alpha.cplusplus.IteratorModeling \
// RUN:   -analyzer-config aggressive-binary-operation-simplification=true \
// RUN:   -analyzer-config c++-container-inlining=false

#include "Inputs/system-header-simulator-cxx.h"

enum Target {MovedFromSource, Destination, storeIterator};

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
    case storeIterator: {
      auto it = l1.begin();
      std::move(it, l1.end(), std::back_inserter(l2));
      return *l1.cbegin(); // expected-warning {{Method called on moved-from object 'l1'}}
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
    case storeIterator: {
      auto it = l1.begin();
      std::move(it, l1.end(), std::back_inserter(l2));
      return *l1.cbegin(); // expected-warning {{Method called on moved-from object 'l1'}}
    }
  }
  return {};
}
