// RUN: %clang_analyze_cc1 -std=c++14 -analyzer-checker=core,debug.ExprInspection -analyzer-config inline-lambdas=true -verify %s
// RUN: %clang_analyze_cc1 -std=c++17 -analyzer-checker=core,debug.ExprInspection -analyzer-config inline-lambdas=true -verify %s

#include "Inputs/system-header-simulator-cxx.h"

void clang_analyzer_warnIfReached();
void clang_analyzer_eval(int);

// Capture copy elided object.
struct Elided{
  int x = 14;
  Elided(int) {}
};

void testCopyElidedObjectCaptured(int x) {
  int r = [e = Elided(x)] {
    return e.x;
  }();
  
  clang_analyzer_eval(r == 14); // expected-warning{{TRUE}}
}

static auto MakeUniquePtr() { return std::make_unique<std::vector<int>>(); }

void testCopyElidedUniquePtr() {
  [uniquePtr = MakeUniquePtr()] {}();
  clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
}
