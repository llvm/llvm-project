// RUN: %clang_analyze_cc1 -std=c++11 -analyzer-checker=core,cplusplus,debug.DebugIteratorModeling,debug.ExprInspection -analyzer-config aggressive-binary-operation-simplification=true -analyzer-config c++-container-inlining=false %s -verify

// RUN: %clang_analyze_cc1 -std=c++11 -analyzer-checker=core,cplusplus,debug.DebugIteratorModeling,debug.ExprInspection -analyzer-config aggressive-binary-operation-simplification=true -analyzer-config c++-container-inlining=true -DINLINE=1 %s -verify

// RUN: %clang_analyze_cc1 -std=c++11 -analyzer-checker=core,cplusplus,alpha.cplusplus.IteratorModeling,debug.ExprInspection -analyzer-config aggressive-binary-operation-simplification=true %s 2>&1 | FileCheck %s

#include "Inputs/system-header-simulator-cxx.h"

template <typename Container>
long clang_analyzer_container_begin(const Container&);
template <typename Container>
long clang_analyzer_container_end(const Container&);

void clang_analyzer_denote(long, const char*);
void clang_analyzer_express(long);
void clang_analyzer_eval(bool);
void clang_analyzer_warnIfReached();

void begin(const std::vector<int> &V) {
  V.begin();

  clang_analyzer_denote(clang_analyzer_container_begin(V), "$V.begin()");
  clang_analyzer_express(clang_analyzer_container_begin(V)); //expected-warning{{$V.begin()}}
}

void end(const std::vector<int> &V) {
  V.end();

  clang_analyzer_denote(clang_analyzer_container_end(V), "$V.end()");
  clang_analyzer_express(clang_analyzer_container_end(V)); //expected-warning{{$V.end()}}
}

////////////////////////////////////////////////////////////////////////////////
///
/// C O N T A I N E R   A S S I G N M E N T S
///
////////////////////////////////////////////////////////////////////////////////

// Move

void move_assignment(std::vector<int> &V1, std::vector<int> &V2) {
  V1.cbegin();
  V1.cend();
  V2.cbegin();
  V2.cend();
  long B1 = clang_analyzer_container_begin(V1);
  long E1 = clang_analyzer_container_end(V1);
  long B2 = clang_analyzer_container_begin(V2);
  long E2 = clang_analyzer_container_end(V2);
  V1 = std::move(V2);
  clang_analyzer_eval(clang_analyzer_container_begin(V1) == B2); //expected-warning{{TRUE}}
  clang_analyzer_eval(clang_analyzer_container_end(V2) == E2); //expected-warning{{TRUE}}
}

////////////////////////////////////////////////////////////////////////////////
///
/// C O N T A I N E R   M O D I F I E R S
///
////////////////////////////////////////////////////////////////////////////////

/// push_back()
///
/// Design decision: extends containers to the ->RIGHT-> (i.e. the
/// past-the-end position of the container is incremented).

void push_back(std::vector<int> &V, int n) {
  V.cbegin();
  V.cend();

  clang_analyzer_denote(clang_analyzer_container_begin(V), "$V.begin()");
  clang_analyzer_denote(clang_analyzer_container_end(V), "$V.end()");

  V.push_back(n);

  clang_analyzer_express(clang_analyzer_container_begin(V)); // expected-warning{{$V.begin()}}
  clang_analyzer_express(clang_analyzer_container_end(V)); // expected-warning{{$V.end() + 1}}
}

/// emplace_back()
///
/// Design decision: extends containers to the ->RIGHT-> (i.e. the
/// past-the-end position of the container is incremented).

void emplace_back(std::vector<int> &V, int n) {
  V.cbegin();
  V.cend();

  clang_analyzer_denote(clang_analyzer_container_begin(V), "$V.begin()");
  clang_analyzer_denote(clang_analyzer_container_end(V), "$V.end()");

  V.emplace_back(n);

  clang_analyzer_express(clang_analyzer_container_begin(V)); // expected-warning{{$V.begin()}}
  clang_analyzer_express(clang_analyzer_container_end(V)); // expected-warning{{$V.end() + 1}}
}

/// pop_back()
///
/// Design decision: shrinks containers to the <-LEFT<- (i.e. the
/// past-the-end position of the container is decremented).

void pop_back(std::vector<int> &V, int n) {
  V.cbegin();
  V.cend();

  clang_analyzer_denote(clang_analyzer_container_begin(V), "$V.begin()");
  clang_analyzer_denote(clang_analyzer_container_end(V), "$V.end()");

  V.pop_back();

  clang_analyzer_express(clang_analyzer_container_begin(V)); // expected-warning{{$V.begin()}}
  clang_analyzer_express(clang_analyzer_container_end(V)); // expected-warning{{$V.end() - 1}}
}

/// push_front()
///
/// Design decision: extends containers to the <-LEFT<- (i.e. the first
/// position of the container is decremented).

void push_front(std::deque<int> &D, int n) {
  D.cbegin();
  D.cend();

  clang_analyzer_denote(clang_analyzer_container_begin(D), "$D.begin()");
  clang_analyzer_denote(clang_analyzer_container_end(D), "$D.end()");

  D.push_front(n);

  clang_analyzer_express(clang_analyzer_container_begin(D)); // expected-warning{{$D.begin()}} FIXME: Should be $D.begin() - 1 (to correctly track the container's size)
  clang_analyzer_express(clang_analyzer_container_end(D)); // expected-warning{{$D.end()}}
}

/// emplace_front()
///
/// Design decision: extends containers to the <-LEFT<- (i.e. the first
/// position of the container is decremented).

void deque_emplace_front(std::deque<int> &D, int n) {
  D.cbegin();
  D.cend();

  clang_analyzer_denote(clang_analyzer_container_begin(D), "$D.begin()");
  clang_analyzer_denote(clang_analyzer_container_end(D), "$D.end()");

  D.emplace_front(n);

  clang_analyzer_express(clang_analyzer_container_begin(D)); // expected-warning{{$D.begin()}} FIXME: Should be $D.begin - 1 (to correctly track the container's size)
  clang_analyzer_express(clang_analyzer_container_end(D)); // expected-warning{{$D.end()}}
}

/// pop_front()
///
/// Design decision: shrinks containers to the ->RIGHT-> (i.e. the first
/// position of the container is incremented).

void deque_pop_front(std::deque<int> &D, int n) {
  D.cbegin();
  D.cend();

  clang_analyzer_denote(clang_analyzer_container_begin(D), "$D.begin()");
  clang_analyzer_denote(clang_analyzer_container_end(D), "$D.end()");

  D.pop_front();

  clang_analyzer_express(clang_analyzer_container_begin(D)); // expected-warning{{$D.begin() + 1}}
  clang_analyzer_express(clang_analyzer_container_end(D)); // expected-warning{{$D.end()}}
}

void clang_analyzer_printState();

void print_state(std::vector<int> &V) {
  V.cbegin();
  clang_analyzer_printState();

// CHECK:      "checker_messages": [
// CHECK-NEXT:   { "checker": "alpha.cplusplus.ContainerModeling", "messages": [
// CHECK-NEXT:     "Container Data :",
// CHECK-NEXT:     "SymRegion{reg_$[[#]]<std::vector<int> & V>} : [ conj_$[[#]]{long, LC[[#]], S[[#]], #[[#]]} .. <Unknown> ]"
// CHECK-NEXT:   ]}

  V.cend();
  clang_analyzer_printState();
  
// CHECK:      "checker_messages": [
// CHECK-NEXT:   { "checker": "alpha.cplusplus.ContainerModeling", "messages": [
// CHECK-NEXT:     "Container Data :",
// CHECK-NEXT:     "SymRegion{reg_$[[#]]<std::vector<int> & V>} : [ conj_$[[#]]{long, LC[[#]], S[[#]], #[[#]]} .. conj_$[[#]]{long, LC[[#]], S[[#]], #[[#]]} ]"
// CHECK-NEXT:   ]}
}
