// RUN: %clang_analyze_cc1 -std=c++11 -analyzer-checker=core,cplusplus,debug.DebugIteratorModeling,debug.ExprInspection -analyzer-config aggressive-binary-operation-simplification=true -analyzer-config c++-container-inlining=false %s -analyzer-output=text -verify

// RUN: %clang_analyze_cc1 -std=c++11 -analyzer-checker=core,cplusplus,debug.DebugIteratorModeling,debug.ExprInspection -analyzer-config aggressive-binary-operation-simplification=true -analyzer-config c++-container-inlining=true -DINLINE=1 %s -analyzer-output=text -verify

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
  clang_analyzer_express(clang_analyzer_container_begin(V)); // expected-warning{{$V.begin()}}
                                                             // expected-note@-1{{$V.begin()}}
}

void end(const std::vector<int> &V) {
  V.end();

  clang_analyzer_denote(clang_analyzer_container_end(V), "$V.end()");
  clang_analyzer_express(clang_analyzer_container_end(V)); // expected-warning{{$V.end()}}
                                                           // expected-note@-1{{$V.end()}}
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
  clang_analyzer_eval(clang_analyzer_container_begin(V1) == B2); // expected-warning{{TRUE}}
                                                                 // expected-note@-1{{TRUE}}
  clang_analyzer_eval(clang_analyzer_container_end(V2) == E2); // expected-warning{{TRUE}}
                                                               // expected-note@-1{{TRUE}}
}

////////////////////////////////////////////////////////////////////////////////
///
/// C O N T A I N E R   M O D I F I E R S
///
////////////////////////////////////////////////////////////////////////////////

/// push_back()
///
/// Design decision: extends containers to the ->BACK-> (i.e. the
/// past-the-end position of the container is incremented).

void clang_analyzer_dump(void*);

void push_back(std::vector<int> &V, int n) {
  V.cbegin();
  V.cend();

  clang_analyzer_denote(clang_analyzer_container_begin(V), "$V.begin()");
  clang_analyzer_denote(clang_analyzer_container_end(V), "$V.end()");

  V.push_back(n); // expected-note 2{{Container 'V' extended to the back by 1 position}}

  clang_analyzer_express(clang_analyzer_container_begin(V)); // expected-warning{{$V.begin()}}
                                                             // expected-note@-1{{$V.begin()}}
  clang_analyzer_express(clang_analyzer_container_end(V)); // expected-warning{{$V.end() + 1}}
                                                           // expected-note@-1{{$V.end() + 1}}
}

/// emplace_back()
///
/// Design decision: extends containers to the ->BACK-> (i.e. the
/// past-the-end position of the container is incremented).

void emplace_back(std::vector<int> &V, int n) {
  V.cbegin();
  V.cend();

  clang_analyzer_denote(clang_analyzer_container_begin(V), "$V.begin()");
  clang_analyzer_denote(clang_analyzer_container_end(V), "$V.end()");

  V.emplace_back(n); // expected-note 2{{Container 'V' extended to the back by 1 position}}

  clang_analyzer_express(clang_analyzer_container_begin(V)); // expected-warning{{$V.begin()}}
                                                             // expected-note@-1{{$V.begin()}}
  clang_analyzer_express(clang_analyzer_container_end(V)); // expected-warning{{$V.end() + 1}}
                                                           // expected-note@-1{{$V.end() + 1}}
}

/// pop_back()
///
/// Design decision: shrinks containers to the <-FRONT<- (i.e. the
/// past-the-end position of the container is decremented).

void pop_back(std::vector<int> &V, int n) {
  V.cbegin();
  V.cend();

  clang_analyzer_denote(clang_analyzer_container_begin(V), "$V.begin()");
  clang_analyzer_denote(clang_analyzer_container_end(V), "$V.end()");

  V.pop_back(); // expected-note 2{{Container 'V' shrank from the back by 1 position}}


  clang_analyzer_express(clang_analyzer_container_begin(V)); // expected-warning{{$V.begin()}}
                                                             // expected-note@-1{{$V.begin()}}
  clang_analyzer_express(clang_analyzer_container_end(V)); // expected-warning{{$V.end() - 1}}
                                                           // expected-note@-1{{$V.end() - 1}}
}

/// push_front()
///
/// Design decision: extends containers to the <-FRONT<- (i.e. the first
/// position of the container is decremented).

void push_front(std::list<int> &L, int n) {
  L.cbegin();
  L.cend();

  clang_analyzer_denote(clang_analyzer_container_begin(L), "$L.begin()");
  clang_analyzer_denote(clang_analyzer_container_end(L), "$L.end()");

  L.push_front(n); // expected-note 2{{Container 'L' extended to the front by 1 position}}

  clang_analyzer_express(clang_analyzer_container_begin(L)); // expected-warning{{$L.begin() - 1}}
                                                             // expected-note@-1{{$L.begin() - 1}}
  clang_analyzer_express(clang_analyzer_container_end(L)); // expected-warning{{$L.end()}}
                                                           // expected-note@-1{{$L.end()}}
}

/// emplace_front()
///
/// Design decision: extends containers to the <-FRONT<- (i.e. the first
/// position of the container is decremented).

void emplace_front(std::list<int> &L, int n) {
  L.cbegin();
  L.cend();

  clang_analyzer_denote(clang_analyzer_container_begin(L), "$L.begin()");
  clang_analyzer_denote(clang_analyzer_container_end(L), "$L.end()");

  L.emplace_front(n); // expected-note 2{{Container 'L' extended to the front by 1 position}}

  clang_analyzer_express(clang_analyzer_container_begin(L)); // expected-warning{{$L.begin() - 1}}
                                                             // expected-note@-1{{$L.begin() - 1}}
  clang_analyzer_express(clang_analyzer_container_end(L)); // expected-warning{{$L.end()}}
                                                           // expected-note@-1{{$L.end()}}
}

/// pop_front()
///
/// Design decision: shrinks containers to the ->BACK-> (i.e. the first
/// position of the container is incremented).

void pop_front(std::list<int> &L, int n) {
  L.cbegin();
  L.cend();

  clang_analyzer_denote(clang_analyzer_container_begin(L), "$L.begin()");
  clang_analyzer_denote(clang_analyzer_container_end(L), "$L.end()");

  L.pop_front(); // expected-note 2{{Container 'L' shrank from the front by 1 position}}

  clang_analyzer_express(clang_analyzer_container_begin(L)); // expected-warning{{$L.begin() + 1}}
                                                             // expected-note@-1{{$L.begin() + 1}}
  clang_analyzer_express(clang_analyzer_container_end(L)); // expected-warning{{$L.end()}}
                                                           // expected-note@-1{{$L.end()}}
}

////////////////////////////////////////////////////////////////////////////////
///
/// O T H E R   T E S T S
///
////////////////////////////////////////////////////////////////////////////////

/// Track local variable

void push_back() {
  std::vector<int> V;
  V.end();
  
  clang_analyzer_denote(clang_analyzer_container_end(V), "$V.end()");

  V.push_back(1); // expected-note{{Container 'V' extended to the back by 1 position}}

  clang_analyzer_express(clang_analyzer_container_end(V)); // expected-warning{{$V.end() + 1}}
                                                           // expected-note@-1{{$V.end() + 1}}
}

/// Track the right container only

void push_back1(std::vector<int> &V1, std::vector<int> &V2, int n) {
  V1.cbegin();
  V1.cend();
  V2.cbegin();
  V2.cend();

  clang_analyzer_denote(clang_analyzer_container_begin(V1), "$V1.begin()");

  V2.push_back(n); // no-note

  clang_analyzer_express(clang_analyzer_container_begin(V1)); // expected-warning{{$V1.begin()}}
                                                              // expected-note@-1{{$V1.begin()}}
}

void push_back2(std::vector<int> &V1, std::vector<int> &V2, int n) {
  V1.cbegin();
  V1.cend();
  V2.cbegin();
  V2.cend();

  clang_analyzer_denote(clang_analyzer_container_begin(V1), "$V1.begin()");
  clang_analyzer_denote(clang_analyzer_container_begin(V2), "$V2.begin()");

  V1.push_back(n); // expected-note{{Container 'V1' extended to the back by 1 position}}
                   // Only once!

  clang_analyzer_express(clang_analyzer_container_begin(V1)); // expected-warning{{$V1.begin()}}
                                                              // expected-note@-1{{$V1.begin()}}

  clang_analyzer_express(clang_analyzer_container_begin(V2)); // expected-warning{{$V2.begin()}}
                                                              // expected-note@-1{{$V2.begin()}}
}

/// Print Container Data as Part of the Program State

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
