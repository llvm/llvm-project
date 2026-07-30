// RUN: %clang_analyze_cc1 -std=c++11\
// RUN: -analyzer-checker=core,cplusplus\
// RUN: -analyzer-checker=debug.DebugContainerModeling,debug.ExprInspection\
// RUN: -analyzer-config aggressive-binary-operation-simplification=true\
// RUN: -analyzer-config c++-container-inlining=false %s -verify

// RUN: %clang_analyze_cc1 -std=c++11\
// RUN: -analyzer-checker=core,cplusplus\
// RUN: -analyzer-checker=debug.DebugContainerModeling,debug.ExprInspection\
// RUN: -analyzer-config aggressive-binary-operation-simplification=true\
// RUN: -analyzer-config c++-container-inlining=true -DINLINE=1 %s -verify

#include "Inputs/system-header-simulator-cxx.h"

template <typename Container>
long clang_analyzer_container_begin(const Container&);
template <typename Container>
long clang_analyzer_container_end(const Container&);
void clang_analyzer_denote(long, const char*);
void clang_analyzer_express(long);

void container_begin_end(const std::vector<int> v0) {
  v0.begin();
  v0.end();

  clang_analyzer_denote(clang_analyzer_container_begin(v0), "$b0");
  clang_analyzer_denote(clang_analyzer_container_end(v0), "$e0");

  clang_analyzer_express(clang_analyzer_container_begin(v0)); // expected-warning{{$b0}}
  clang_analyzer_express(clang_analyzer_container_end(v0)); // expected-warning{{$e0}}
}
