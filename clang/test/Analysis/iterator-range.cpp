// RUN: %clang_analyze_cc1 -std=c++11 -analyzer-checker=core,cplusplus,alpha.cplusplus.IteratorRange -analyzer-eagerly-assume -analyzer-config c++-container-inlining=false %s -verify
// RUN: %clang_analyze_cc1 -std=c++11 -analyzer-checker=core,cplusplus,alpha.cplusplus.IteratorRange -analyzer-eagerly-assume -analyzer-config c++-container-inlining=true -DINLINE=1 %s -verify

#include "Inputs/system-header-simulator-cxx.h"

void clang_analyzer_warnIfReached();

void simple_good_end(const std::vector<int> &v) {
  auto i = v.end();
  if (i != v.end()) {
    clang_analyzer_warnIfReached();
    *i; // no-warning
  }
}

void simple_bad_end(const std::vector<int> &v) {
  auto i = v.end();
  *i; // expected-warning{{Iterator accessed outside of its range}}
}
