// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

template<class T> class vector {};
@protocol P @end

// expected-no-diagnostics

vector<id<P>> v;
vector<vector<id<P>>> v2;
