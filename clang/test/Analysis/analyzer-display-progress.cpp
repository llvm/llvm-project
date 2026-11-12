// RUN: %clang_analyze_cc1 -verify %s 2>&1 \
// RUN:   -analyzer-display-progress \
// RUN:   -analyzer-checker=debug.ExprInspection \
// RUN:   -analyzer-output=text \
// RUN: | FileCheck %s

void clang_analyzer_warnIfReached();

// expected-note@+2 {{[debug] analyzing from f()}}
// expected-warning@+1 {{REACHABLE}} expected-note@+1 {{REACHABLE}}
void f() { clang_analyzer_warnIfReached(); }

// expected-note@+2 {{[debug] analyzing from g()}}
// expected-warning@+1 {{REACHABLE}} expected-note@+1 {{REACHABLE}}
void g() { clang_analyzer_warnIfReached(); }

// expected-note@+2 {{[debug] analyzing from h()}}
// expected-warning@+1 {{REACHABLE}} expected-note@+1 {{REACHABLE}}
void h() { clang_analyzer_warnIfReached(); }

struct SomeStruct {
  // expected-note@+2 {{[debug] analyzing from SomeStruct::f()}}
  // expected-warning@+1 {{REACHABLE}} expected-note@+1 {{REACHABLE}}
  void f() { clang_analyzer_warnIfReached(); }
};

struct SomeOtherStruct {
  // expected-note@+2 {{[debug] analyzing from SomeOtherStruct::f()}}
  // expected-warning@+1 {{REACHABLE}} expected-note@+1 {{REACHABLE}}
  void f() { clang_analyzer_warnIfReached(); }
};

namespace ns {
  struct SomeStruct {
    // expected-note@+2 {{[debug] analyzing from ns::SomeStruct::f(int)}}
    // expected-warning@+1 {{REACHABLE}} expected-note@+1 {{REACHABLE}}
    void f(int) { clang_analyzer_warnIfReached(); }
    // expected-note@+2 {{[debug] analyzing from ns::SomeStruct::f(float, ::SomeStruct)}}
    // expected-warning@+1 {{REACHABLE}} expected-note@+1 {{REACHABLE}}
    void f(float, ::SomeStruct) { clang_analyzer_warnIfReached(); }
    // expected-note@+2 {{[debug] analyzing from ns::SomeStruct::f(float, SomeStruct)}}
    // expected-warning@+1 {{REACHABLE}} expected-note@+1 {{REACHABLE}}
    void f(float, SomeStruct) { clang_analyzer_warnIfReached(); }
  };
}

// CHECK: analyzer-display-progress.cpp f() : {{[0-9]+}}
// CHECK: analyzer-display-progress.cpp g() : {{[0-9]+}}
// CHECK: analyzer-display-progress.cpp h() : {{[0-9]+}}
// CHECK: analyzer-display-progress.cpp SomeStruct::f() : {{[0-9]+}}
// CHECK: analyzer-display-progress.cpp SomeOtherStruct::f() : {{[0-9]+}}
// CHECK: analyzer-display-progress.cpp ns::SomeStruct::f(int) : {{[0-9]+}}
// CHECK: analyzer-display-progress.cpp ns::SomeStruct::f(float, ::SomeStruct) : {{[0-9]+}}
// CHECK: analyzer-display-progress.cpp ns::SomeStruct::f(float, SomeStruct) : {{[0-9]+}}
