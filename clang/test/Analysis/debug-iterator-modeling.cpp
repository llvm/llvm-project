// RUN: %clang_analyze_cc1 -std=c++11\
// RUN: -analyzer-checker=core,cplusplus\
// RUN: -analyzer-checker=debug.DebugIteratorModeling,debug.ExprInspection\
// RUN: -analyzer-config aggressive-binary-operation-simplification=true\
// RUN: -analyzer-config c++-container-inlining=false %s -verify

// RUN: %clang_analyze_cc1 -std=c++11\
// RUN: -analyzer-checker=core,cplusplus\
// RUN: -analyzer-checker=debug.DebugIteratorModeling,debug.ExprInspection\
// RUN: -analyzer-config aggressive-binary-operation-simplification=true\
// RUN: -analyzer-config c++-container-inlining=true -DINLINE=1 %s -verify

#include "Inputs/system-header-simulator-cxx.h"

template <typename Container>
long clang_analyzer_container_begin(const Container&);
template <typename Container>
long clang_analyzer_container_end(const Container&);
template <typename Iterator>
long clang_analyzer_iterator_position(const Iterator&);
template <typename Iterator>
void* clang_analyzer_iterator_container(const Iterator&);
template <typename Iterator>
bool clang_analyzer_iterator_validity(const Iterator&);
void clang_analyzer_denote(long, const char*);
void clang_analyzer_express(long);
void clang_analyzer_dump(const void*);
void clang_analyzer_eval(bool);

void iterator_position(const std::vector<int> v0) {
  auto b0 = v0.begin(), e0 = v0.end();

  clang_analyzer_denote(clang_analyzer_container_begin(v0), "$b0");
  clang_analyzer_denote(clang_analyzer_container_end(v0), "$e0");

  clang_analyzer_express(clang_analyzer_iterator_position(b0)); // expected-warning{{$b0}}
  clang_analyzer_express(clang_analyzer_iterator_position(e0)); // expected-warning{{$e0}}

  ++b0;

  clang_analyzer_express(clang_analyzer_iterator_position(b0)); // expected-warning{{$b0 + 1}}
}

void iterator_container(const std::vector<int> v0) {
  auto b0 = v0.begin();

  clang_analyzer_dump(&v0); //expected-warning{{&v0}}
  clang_analyzer_eval(clang_analyzer_iterator_container(b0) == &v0); // expected-warning{{TRUE}}
}

void iterator_validity(std::vector<int> v0) {
  auto b0 = v0.begin();
  clang_analyzer_eval(clang_analyzer_iterator_validity(b0)); //expected-warning{{TRUE}}

  v0.clear();

  clang_analyzer_eval(clang_analyzer_iterator_validity(b0)); //expected-warning{{FALSE}}
}
