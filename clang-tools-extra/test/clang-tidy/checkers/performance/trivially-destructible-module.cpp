// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
// RUN: %clang -std=c++20 -x c++-module --precompile %t/mymodule.cppm -o %t/mymodule.pcm -I%t
// RUN: %check_clang_tidy -std=c++20 %t/main.cpp performance-trivially-destructible %t/out \
// RUN:     -- -- -std=c++20 -I%t -fmodule-file=mymodule=%t/mymodule.pcm

// Test: C++20 modules - class visible through both #include and import.
//
// When a class is defined in a header that is both #include'd directly and
// included in a module's global module fragment, the class's destructor may
// appear multiple times in the redeclaration chain.
//
// This test verifies:
// 1. No false positive on in-line defaulted destructor (struct A)
// 2. No false positive on implicit destructor (struct X)
// 3. Correct warning on out-of-line defaulted destructor (struct B)
//
// We expect exactly 1 warning (for struct B's out-of-line destructor).

//--- header.h
#pragma once

// Negative cases: with modules, the destructor's redeclaration chain may
// contain multiple entries, which can trigger false positives if not handled
// correctly.

// In-line defaulted destructor should NOT warn
struct A {
    ~A() = default;
};

// Implicit destructor should NOT warn
struct X {
    A a;
};

// Positive case: out-of-line defaulted destructor SHOULD warn
struct B {
    ~B();
    int x;
};

//--- mymodule.cppm
module;
#include "header.h"
export module mymodule;

//--- main.cpp
#include "header.h"
import mymodule;

// CHECK-MESSAGES: header.h:19:5: warning: class 'B' can be made trivially destructible
B::~B() = default; // to-be-removed
// CHECK-MESSAGES: :[[@LINE-1]]:4: note: destructor definition is here
// CHECK-FIXES: // to-be-removed

int main() {
    X x;
    B b;
}
