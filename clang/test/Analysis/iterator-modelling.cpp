// RUN: %clang_analyze_cc1 -std=c++11 -analyzer-checker=core,cplusplus,debug.DebugIteratorModeling,debug.ExprInspection -analyzer-config aggressive-binary-operation-simplification=true -analyzer-config c++-container-inlining=false %s -verify

// RUN: %clang_analyze_cc1 -std=c++11 -analyzer-checker=core,cplusplus,debug.DebugIteratorModeling,debug.ExprInspection -analyzer-config aggressive-binary-operation-simplification=true -analyzer-config c++-container-inlining=true -DINLINE=1 %s -verify

// RUN: %clang_analyze_cc1 -std=c++11 -analyzer-checker=core,cplusplus,alpha.cplusplus.IteratorModeling,debug.ExprInspection -analyzer-config aggressive-binary-operation-simplification=true %s 2>&1 | FileCheck %s

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
void clang_analyzer_eval(bool);
void clang_analyzer_warnIfReached();

void begin(const std::vector<int> &v) {
  auto i = v.begin();

  clang_analyzer_eval(clang_analyzer_iterator_container(i) == &v); // expected-warning{{TRUE}}
  clang_analyzer_denote(clang_analyzer_container_begin(v), "$v.begin()");
  clang_analyzer_express(clang_analyzer_iterator_position(i)); //expected-warning{{$v.begin()}}

  if (i != v.begin()) {
    clang_analyzer_warnIfReached();
  }
}

void end(const std::vector<int> &v) {
  auto i = v.end();

  clang_analyzer_eval(clang_analyzer_iterator_container(i) == &v); // expected-warning{{TRUE}}
  clang_analyzer_denote(clang_analyzer_container_end(v), "$v.end()");
  clang_analyzer_express(clang_analyzer_iterator_position(i)); //expected-warning{{$v.end()}}

  if (i != v.end()) {
    clang_analyzer_warnIfReached();
  }
}

void prefix_increment(const std::vector<int> &v) {
  auto i = v.begin();

  clang_analyzer_denote(clang_analyzer_container_begin(v), "$v.begin()");

  auto j = ++i;

  clang_analyzer_express(clang_analyzer_iterator_position(i)); //expected-warning{{$v.begin() + 1}}
  clang_analyzer_express(clang_analyzer_iterator_position(j)); //expected-warning{{$v.begin() + 1}}
}

void prefix_decrement(const std::vector<int> &v) {
  auto i = v.end();

  clang_analyzer_denote(clang_analyzer_container_end(v), "$v.end()");

  auto j = --i;

  clang_analyzer_express(clang_analyzer_iterator_position(i)); //expected-warning{{$v.end() - 1}}
  clang_analyzer_express(clang_analyzer_iterator_position(j)); //expected-warning{{$v.end() - 1}}
}

void postfix_increment(const std::vector<int> &v) {
  auto i = v.begin();

  clang_analyzer_denote(clang_analyzer_container_begin(v), "$v.begin()");

  auto j = i++;

  clang_analyzer_express(clang_analyzer_iterator_position(i)); //expected-warning{{$v.begin() + 1}}
  clang_analyzer_express(clang_analyzer_iterator_position(j)); //expected-warning{{$v.begin()}}
}

void postfix_decrement(const std::vector<int> &v) {
  auto i = v.end();

  clang_analyzer_denote(clang_analyzer_container_end(v), "$v.end()");

  auto j = i--;

  clang_analyzer_express(clang_analyzer_iterator_position(i)); //expected-warning{{$v.end() - 1}}
  clang_analyzer_express(clang_analyzer_iterator_position(j)); //expected-warning{{$v.end()}}
}

void plus_equal(const std::vector<int> &v) {
  auto i = v.begin();

  clang_analyzer_denote(clang_analyzer_container_begin(v), "$v.begin()");

  i += 2;

  clang_analyzer_express(clang_analyzer_iterator_position(i)); //expected-warning{{$v.begin() + 2}}
}

void minus_equal(const std::vector<int> &v) {
  auto i = v.end();

  clang_analyzer_denote(clang_analyzer_container_end(v), "$v.end()");

  i -= 2;

  clang_analyzer_express(clang_analyzer_iterator_position(i)); //expected-warning{{$v.end() - 2}}
}

void copy(const std::vector<int> &v) {
  auto i1 = v.end();

  clang_analyzer_denote(clang_analyzer_container_end(v), "$v.end()");

  auto i2 = i1;

  clang_analyzer_eval(clang_analyzer_iterator_container(i2) == &v); // expected-warning{{TRUE}}
  clang_analyzer_express(clang_analyzer_iterator_position(i2)); //expected-warning{{$v.end()}}
}

void plus(const std::vector<int> &v) {
  auto i1 = v.begin();

  clang_analyzer_denote(clang_analyzer_container_begin(v), "$v.begin()");

  auto i2 = i1 + 2;

  clang_analyzer_eval(clang_analyzer_iterator_container(i2) == &v); // expected-warning{{TRUE}}
  clang_analyzer_express(clang_analyzer_iterator_position(i2)); //expected-warning{{$v.begin() + 2}}
}

void minus(const std::vector<int> &v) {
  auto i1 = v.end();

  clang_analyzer_denote(clang_analyzer_container_end(v), "$v.end()");

  auto i2 = i1 - 2;

  clang_analyzer_eval(clang_analyzer_iterator_container(i2) == &v); // expected-warning{{TRUE}}
  clang_analyzer_express(clang_analyzer_iterator_position(i2)); //expected-warning{{$v.end() - 2}}
}

void copy_and_increment1(const std::vector<int> &v) {
  auto i1 = v.begin();

  clang_analyzer_denote(clang_analyzer_container_begin(v), "$v.begin()");

  auto i2 = i1;
  ++i1;

  clang_analyzer_express(clang_analyzer_iterator_position(i1)); //expected-warning{{$v.begin() + 1}}
  clang_analyzer_express(clang_analyzer_iterator_position(i2)); //expected-warning{{$v.begin()}}
}

void copy_and_increment2(const std::vector<int> &v) {
  auto i1 = v.begin();

  clang_analyzer_denote(clang_analyzer_container_begin(v), "$v.begin()");

  auto i2 = i1;
  ++i2;

  clang_analyzer_express(clang_analyzer_iterator_position(i1)); //expected-warning{{$v.begin()}}
  clang_analyzer_express(clang_analyzer_iterator_position(i2)); //expected-warning{{$v.begin() + 1}}
}

void copy_and_decrement1(const std::vector<int> &v) {
  auto i1 = v.end();

  clang_analyzer_denote(clang_analyzer_container_end(v), "$v.end()");

  auto i2 = i1;
  --i1;

  clang_analyzer_express(clang_analyzer_iterator_position(i1)); //expected-warning{{$v.end() - 1}}
  clang_analyzer_express(clang_analyzer_iterator_position(i2)); //expected-warning{{$v.end()}}
}

void copy_and_decrement2(const std::vector<int> &v) {
  auto i1 = v.end();

  clang_analyzer_denote(clang_analyzer_container_end(v), "$v.end()");

  auto i2 = i1;
  --i2;

  clang_analyzer_express(clang_analyzer_iterator_position(i1)); //expected-warning{{$v.end()}}
  clang_analyzer_express(clang_analyzer_iterator_position(i2)); //expected-warning{{$v.end() - 1}}
}

////////////////////////////////////////////////////////////////////////////////
///
/// C O N T A I N E R   A S S I G N M E N T S
///
////////////////////////////////////////////////////////////////////////////////

// Copy

void list_copy_assignment(std::list<int> &L1, const std::list<int> &L2) {
  auto i0 = L1.cbegin();
  clang_analyzer_eval(clang_analyzer_iterator_validity(i0)); //expected-warning{{TRUE}}
  L1 = L2;
  clang_analyzer_eval(clang_analyzer_iterator_validity(i0)); //expected-warning{{FALSE}}
}

void vector_copy_assignment(std::vector<int> &V1, const std::vector<int> &V2) {
  auto i0 = V1.cbegin();
  clang_analyzer_eval(clang_analyzer_iterator_validity(i0)); //expected-warning{{TRUE}}
  V1 = V2;
  clang_analyzer_eval(clang_analyzer_iterator_validity(i0)); //expected-warning{{FALSE}}
}

void deque_copy_assignment(std::deque<int> &D1, const std::deque<int> &D2) {
  auto i0 = D1.cbegin();
  clang_analyzer_eval(clang_analyzer_iterator_validity(i0)); //expected-warning{{TRUE}}
  D1 = D2;
  clang_analyzer_eval(clang_analyzer_iterator_validity(i0)); //expected-warning{{FALSE}}
}

void forward_list_copy_assignment(std::forward_list<int> &FL1,
                                  const std::forward_list<int> &FL2) {
  auto i0 = FL1.cbegin();
  clang_analyzer_eval(clang_analyzer_iterator_validity(i0)); //expected-warning{{TRUE}}
  FL1 = FL2;
  clang_analyzer_eval(clang_analyzer_iterator_validity(i0)); //expected-warning{{FALSE}}
}

// Move

void list_move_assignment(std::list<int> &L1, std::list<int> &L2) {
  auto i0 = L1.cbegin(), i1 = L2.cbegin(), i2 = --L2.cend(), i3 = L2.cend();

  clang_analyzer_denote(clang_analyzer_container_begin(L2), "$L2.begin()");
  clang_analyzer_denote(clang_analyzer_container_end(L2), "$L2.end()");

  L1 = std::move(L2);

  clang_analyzer_eval(clang_analyzer_iterator_validity(i0)); //expected-warning{{FALSE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i1)); //expected-warning{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i2)); //expected-warning{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i3)); //expected-warning{{TRUE}} FIXME: Should be FALSE.

  clang_analyzer_eval(clang_analyzer_iterator_container(i1) == &L1); // expected-warning{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_container(i2) == &L1); // expected-warning{{TRUE}}

  clang_analyzer_express(clang_analyzer_container_begin(L1)); // expected-warning{{$L2.begin()}}
  clang_analyzer_express(clang_analyzer_iterator_position(i1)); // expected-warning{{$L2.begin()}}
}

void vector_move_assignment(std::vector<int> &V1, std::vector<int> &V2) {
  auto i0 = V1.cbegin(), i1 = V2.cbegin(), i2 = --V2.cend(), i3 = V2.cend();

  clang_analyzer_denote(clang_analyzer_container_begin(V2), "$V2.begin()");

  V1 = std::move(V2);

  clang_analyzer_eval(clang_analyzer_iterator_validity(i0)); //expected-warning{{FALSE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i1)); //expected-warning{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i2)); //expected-warning{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i3)); //expected-warning{{TRUE}} FIXME: Should be FALSE.

  clang_analyzer_eval(clang_analyzer_iterator_container(i1) == &V1); // expected-warning{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_container(i2) == &V1); // expected-warning{{TRUE}}

  clang_analyzer_express(clang_analyzer_container_begin(V1)); // expected-warning{{$V2.begin()}}
  clang_analyzer_express(clang_analyzer_iterator_position(i1)); // expected-warning{{$V2.begin()}}
}

void deque_move_assignment(std::deque<int> &D1, std::deque<int> &D2) {
  auto i0 = D1.cbegin(), i1 = D2.cbegin(), i2 = --D2.cend(), i3 = D2.cend();

  clang_analyzer_denote(clang_analyzer_container_begin(D2), "$D2.begin()");

  D1 = std::move(D2);

  clang_analyzer_eval(clang_analyzer_iterator_validity(i0)); //expected-warning{{FALSE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i1)); //expected-warning{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i2)); //expected-warning{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i3)); //expected-warning{{TRUE}} FIXME: Should be FALSE.

  clang_analyzer_eval(clang_analyzer_iterator_container(i1) == &D1); // expected-warning{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_container(i2) == &D1); // expected-warning{{TRUE}}

  clang_analyzer_express(clang_analyzer_container_begin(D1)); // expected-warning{{$D2.begin()}}
  clang_analyzer_express(clang_analyzer_iterator_position(i1)); // expected-warning{{$D2.begin()}}
}

void forward_list_move_assignment(std::forward_list<int> &FL1,
                                  std::forward_list<int> &FL2) {
  auto i0 = FL1.cbegin(), i1 = FL2.cbegin(), i2 = FL2.cend();

  clang_analyzer_denote(clang_analyzer_container_begin(FL2), "$FL2.begin()");

  FL1 = std::move(FL2);

  clang_analyzer_eval(clang_analyzer_iterator_validity(i0)); //expected-warning{{FALSE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i1)); //expected-warning{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i2)); //expected-warning{{TRUE}} FIXME: Should be FALSE.

  clang_analyzer_eval(clang_analyzer_iterator_container(i1) == &FL1); // expected-warning{{TRUE}}

  clang_analyzer_express(clang_analyzer_container_begin(FL1)); // expected-warning{{$FL2.begin()}}
  clang_analyzer_express(clang_analyzer_iterator_position(i1)); // expected-warning{{$FL2.begin()}}
}


////////////////////////////////////////////////////////////////////////////////
///
/// C O N T A I N E R   M O D I F I E R S
///
////////////////////////////////////////////////////////////////////////////////

/// assign()
///
/// - Invalidates all iterators, including the past-the-end iterator for all
///   container types.

void list_assign(std::list<int> &L, int n) {
  auto i0 = L.cbegin(), i1 = L.cend();
  L.assign(10, n);
  clang_analyzer_eval(clang_analyzer_iterator_validity(i0)); //expected-warning{{FALSE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i1)); //expected-warning{{FALSE}}
}

void vector_assign(std::vector<int> &V, int n) {
  auto i0 = V.cbegin(), i1 = V.cend();
  V.assign(10, n);
  clang_analyzer_eval(clang_analyzer_iterator_validity(i0)); //expected-warning{{FALSE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i1)); //expected-warning{{FALSE}}
}

void deque_assign(std::deque<int> &D, int n) {
  auto i0 = D.cbegin(), i1 = D.cend();
  D.assign(10, n);
  clang_analyzer_eval(clang_analyzer_iterator_validity(i0)); //expected-warning{{FALSE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i1)); //expected-warning{{FALSE}}
}

void forward_list_assign(std::forward_list<int> &FL, int n) {
  auto i0 = FL.cbegin(), i1 = FL.cend();
  FL.assign(10, n);
  clang_analyzer_eval(clang_analyzer_iterator_validity(i0)); //expected-warning{{FALSE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i1)); //expected-warning{{FALSE}}
}

/// clear()
///
/// - Invalidates all iterators, including the past-the-end iterator for all
///   container types.

void list_clear(std::list<int> &L) {
  auto i0 = L.cbegin(), i1 = L.cend();
  clang_analyzer_eval(clang_analyzer_iterator_validity(i0)); //expected-warning{{TRUE}}
  L.clear();
  clang_analyzer_eval(clang_analyzer_iterator_validity(i0)); //expected-warning{{FALSE}}
}

void vector_clear(std::vector<int> &V) {
  auto i0 = V.cbegin(), i1 = V.cend();
  clang_analyzer_eval(clang_analyzer_iterator_validity(i0)); //expected-warning{{TRUE}}
  V.clear();
  clang_analyzer_eval(clang_analyzer_iterator_validity(i0)); //expected-warning{{FALSE}}
}

void deque_clear(std::deque<int> &D) {
  auto i0 = D.cbegin(), i1 = D.cend();
  clang_analyzer_eval(clang_analyzer_iterator_validity(i0)); //expected-warning{{TRUE}}
  D.clear();
  clang_analyzer_eval(clang_analyzer_iterator_validity(i0)); //expected-warning{{FALSE}}
}

void forward_list_clear(std::forward_list<int> &FL) {
  auto i0 = FL.cbegin(), i1 = FL.cend();
  clang_analyzer_eval(clang_analyzer_iterator_validity(i0)); //expected-warning{{TRUE}}
  FL.clear();
  clang_analyzer_eval(clang_analyzer_iterator_validity(i0)); //expected-warning{{FALSE}}
}

/// push_back()
///
/// - Design decision: extends containers to the ->RIGHT-> (i.e. the
///   past-the-end position of the container is incremented).
///
/// - Iterator invalidation rules depend the container type.

/// std::list-like containers: No iterators are invalidated.

void list_push_back(std::list<int> &L, int n) {
  auto i0 = L.cbegin(), i1 = --L.cend(), i2 = L.cend();

  clang_analyzer_denote(clang_analyzer_container_begin(L), "$L.begin()");
  clang_analyzer_denote(clang_analyzer_container_end(L), "$L.end()");

  L.push_back(n);

  clang_analyzer_eval(clang_analyzer_iterator_validity(i0)); //expected-warning{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i1)); //expected-warning{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i2)); //expected-warning{{TRUE}}

  clang_analyzer_express(clang_analyzer_container_begin(L)); // expected-warning{{$L.begin()}}
  clang_analyzer_express(clang_analyzer_iterator_position(i0)); // expected-warning{{$L.begin()}}

  clang_analyzer_express(clang_analyzer_container_end(L)); // expected-warning{{$L.end() + 1}}
  clang_analyzer_express(clang_analyzer_iterator_position(i1)); // expected-warning{{$L.end() - 1}}
  clang_analyzer_express(clang_analyzer_iterator_position(i2)); // expected-warning{{$L.end()}} FIXME: Should be $L.end() + 1
}

/// std::vector-like containers: The past-the-end iterator is invalidated.

void vector_push_back(std::vector<int> &V, int n) {
  auto i0 = V.cbegin(), i1 = --V.cend(), i2 = V.cend();

  clang_analyzer_denote(clang_analyzer_container_begin(V), "$V.begin()");
  clang_analyzer_denote(clang_analyzer_container_end(V), "$V.end()");

  V.push_back(n);

  clang_analyzer_eval(clang_analyzer_iterator_validity(i0)); //expected-warning{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i1)); //expected-warning{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i2)); //expected-warning{{FALSE}}

  clang_analyzer_express(clang_analyzer_container_begin(V)); // expected-warning{{$V.begin()}}
  clang_analyzer_express(clang_analyzer_iterator_position(i0)); // expected-warning{{$V.begin()}}

  clang_analyzer_express(clang_analyzer_container_end(V)); // expected-warning{{$V.end() + 1}}
  clang_analyzer_express(clang_analyzer_iterator_position(i1)); // expected-warning{{$V.end() - 1}}
}

/// std::deque-like containers: All iterators, including the past-the-end
///                             iterator, are invalidated.

void deque_push_back(std::deque<int> &D, int n) {
  auto i0 = D.cbegin(), i1 = --D.cend(), i2 = D.cend();

  clang_analyzer_denote(clang_analyzer_container_begin(D), "$D.begin()");
  clang_analyzer_denote(clang_analyzer_container_end(D), "$D.end()");

  D.push_back(n);

  clang_analyzer_eval(clang_analyzer_iterator_validity(i0)); //expected-warning{{FALSE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i1)); //expected-warning{{FALSE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i2)); //expected-warning{{FALSE}}

  clang_analyzer_express(clang_analyzer_container_begin(D)); // expected-warning{{$D.begin()}}
  clang_analyzer_express(clang_analyzer_container_end(D)); // expected-warning{{$D.end()}} FIXME: Should be $D.end() + 1 (to correctly track the container's size)
}

/// emplace_back()
///
/// - Design decision: extends containers to the ->RIGHT-> (i.e. the
///   past-the-end position of the container is incremented).
///
/// - Iterator invalidation rules depend the container type.

/// std::list-like containers: No iterators are invalidated.

void list_emplace_back(std::list<int> &L, int n) {
  auto i0 = L.cbegin(), i1 = --L.cend(), i2 = L.cend();

  clang_analyzer_denote(clang_analyzer_container_begin(L), "$L.begin()");
  clang_analyzer_denote(clang_analyzer_container_end(L), "$L.end()");

  L.emplace_back(n);

  clang_analyzer_eval(clang_analyzer_iterator_validity(i0)); //expected-warning{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i1)); //expected-warning{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i2)); //expected-warning{{TRUE}}

  clang_analyzer_express(clang_analyzer_container_begin(L)); // expected-warning{{$L.begin()}}
  clang_analyzer_express(clang_analyzer_iterator_position(i0)); // expected-warning{{$L.begin()}}

  clang_analyzer_express(clang_analyzer_container_end(L)); // expected-warning{{$L.end() + 1}}
  clang_analyzer_express(clang_analyzer_iterator_position(i1)); // expected-warning{{$L.end()}}
  clang_analyzer_express(clang_analyzer_iterator_position(i2)); // expected-warning{{$L.end()}}  FIXME: Should be $L.end() + 1
}

/// std::vector-like containers: The past-the-end iterator is invalidated.

void vector_emplace_back(std::vector<int> &V, int n) {
  auto i0 = V.cbegin(), i1 = --V.cend(), i2 = V.cend();

  clang_analyzer_denote(clang_analyzer_container_begin(V), "$V.begin()");
  clang_analyzer_denote(clang_analyzer_container_end(V), "$V.end()");

  V.emplace_back(n);

  clang_analyzer_eval(clang_analyzer_iterator_validity(i0)); //expected-warning{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i1)); //expected-warning{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i2)); //expected-warning{{FALSE}}

  clang_analyzer_express(clang_analyzer_container_begin(V)); // expected-warning{{$V.begin()}}
  clang_analyzer_express(clang_analyzer_iterator_position(i0)); // expected-warning{{$V.begin()}}

  clang_analyzer_express(clang_analyzer_container_end(V)); // expected-warning{{$V.end() + 1}}
  clang_analyzer_express(clang_analyzer_iterator_position(i1)); // expected-warning{{$V.end() - 1}}
}

/// std::deque-like containers: All iterators, including the past-the-end
///                             iterator, are invalidated.

void deque_emplace_back(std::deque<int> &D, int n) {
  auto i0 = D.cbegin(), i1 = --D.cend(), i2 = D.cend();

  clang_analyzer_denote(clang_analyzer_container_begin(D), "$D.begin()");
  clang_analyzer_denote(clang_analyzer_container_end(D), "$D.end()");

  D.emplace_back(n);

  clang_analyzer_eval(clang_analyzer_iterator_validity(i0)); //expected-warning{{FALSE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i1)); //expected-warning{{FALSE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i2)); //expected-warning{{FALSE}}

  clang_analyzer_express(clang_analyzer_container_begin(D)); // expected-warning{{$D.begin()}}
  clang_analyzer_express(clang_analyzer_container_end(D)); // expected-warning{{$D.end()}}  FIXME: Should be $D.end() + 1 (to correctly track the container's size)
}

/// pop_back()
///
/// - Design decision: shrinks containers to the <-LEFT<- (i.e. the
///   past-the-end position of the container is decremented).
///
/// - Iterator invalidation rules depend the container type.

/// std::list-like containers: Iterators to the last element are invalidated.

void list_pop_back(std::list<int> &L, int n) {
  auto i0 = L.cbegin(), i1 = --L.cend(), i2 = L.cend();

  clang_analyzer_denote(clang_analyzer_container_begin(L), "$L.begin()");
  clang_analyzer_denote(clang_analyzer_container_end(L), "$L.end()");

  L.pop_back();

  clang_analyzer_eval(clang_analyzer_iterator_validity(i0)); //expected-warning{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i1)); //expected-warning{{FALSE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i2)); //expected-warning{{TRUE}}

  clang_analyzer_express(clang_analyzer_container_begin(L)); // expected-warning{{$L.begin()}}
  clang_analyzer_express(clang_analyzer_iterator_position(i0)); // expected-warning{{$L.begin()}}

  clang_analyzer_express(clang_analyzer_container_end(L)); // expected-warning{{$L.end() - 1}}
  clang_analyzer_express(clang_analyzer_iterator_position(i2)); // expected-warning{{$L.end()}}  FIXME: Should be $L.end() - 1
}

/// std::vector-like containers: Iterators to the last element, as well as the
///                              past-the-end iterator, are invalidated.

void vector_pop_back(std::vector<int> &V, int n) {
  auto i0 = V.cbegin(), i1 = --V.cend(), i2 = V.cend();

  clang_analyzer_denote(clang_analyzer_container_begin(V), "$V.begin()");
  clang_analyzer_denote(clang_analyzer_container_end(V), "$V.end()");

  V.pop_back();

  clang_analyzer_eval(clang_analyzer_iterator_validity(i0)); //expected-warning{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i1)); //expected-warning{{FALSE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i2)); //expected-warning{{FALSE}}

  clang_analyzer_express(clang_analyzer_container_begin(V)); // expected-warning{{$V.begin()}}
  clang_analyzer_express(clang_analyzer_iterator_position(i0)); // expected-warning{{$V.begin()}}

  clang_analyzer_express(clang_analyzer_container_end(V)); // expected-warning{{$V.end() - 1}}
}

/// std::deque-like containers: Iterators to the last element are invalidated.
///                             The past-the-end iterator is also invalidated.
///                             Other iterators are not affected.

void deque_pop_back(std::deque<int> &D, int n) {
  auto i0 = D.cbegin(), i1 = --D.cend(), i2 = D.cend();

  clang_analyzer_denote(clang_analyzer_container_begin(D), "$D.begin()");
  clang_analyzer_denote(clang_analyzer_container_end(D), "$D.end()");

  D.pop_back();

  clang_analyzer_eval(clang_analyzer_iterator_validity(i0)); //expected-warning{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i1)); //expected-warning{{FALSE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i2)); //expected-warning{{FALSE}}

  clang_analyzer_express(clang_analyzer_container_begin(D)); // expected-warning{{$D.begin()}}
  clang_analyzer_express(clang_analyzer_iterator_position(i0)); // expected-warning{{$D.begin()}}

  clang_analyzer_express(clang_analyzer_container_end(D)); // expected-warning{{$D.end() - 1}}
}

/// push_front()
///
/// - Design decision: extends containers to the <-LEFT<- (i.e. the first
///                    position of the container is decremented).
///
/// - Iterator invalidation rules depend the container type.

/// std::list-like containers: No iterators are invalidated.

void list_push_front(std::list<int> &L, int n) {
  auto i0 = L.cbegin(), i1 = L.cend();

  clang_analyzer_denote(clang_analyzer_container_begin(L), "$L.begin()");
  clang_analyzer_denote(clang_analyzer_container_end(L), "$L.end()");

  L.push_front(n);

  clang_analyzer_eval(clang_analyzer_iterator_validity(i0)); //expected-warning{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i1)); //expected-warning{{TRUE}}

  clang_analyzer_express(clang_analyzer_container_begin(L)); // expected-warning{{$L.begin() - 1}}
  clang_analyzer_express(clang_analyzer_iterator_position(i0)); // expected-warning{{$L.begin()}}

  clang_analyzer_express(clang_analyzer_container_end(L)); // expected-warning{{$L.end()}}
  clang_analyzer_express(clang_analyzer_iterator_position(i1)); // expected-warning{{$L.end()}}
}

/// std::deque-like containers: All iterators, including the past-the-end
///                             iterator, are invalidated.

void deque_push_front(std::deque<int> &D, int n) {
  auto i0 = D.cbegin(), i1 = --D.cend(), i2 = D.cend();

  clang_analyzer_denote(clang_analyzer_container_begin(D), "$D.begin()");
  clang_analyzer_denote(clang_analyzer_container_end(D), "$D.end()");

  D.push_front(n);

  clang_analyzer_eval(clang_analyzer_iterator_validity(i0)); //expected-warning{{FALSE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i1)); //expected-warning{{FALSE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i2)); //expected-warning{{FALSE}}

  clang_analyzer_express(clang_analyzer_container_begin(D)); // expected-warning{{$D.begin()}} FIXME: Should be $D.begin() - 1 (to correctly track the container's size)

  clang_analyzer_express(clang_analyzer_container_end(D)); // expected-warning{{$D.end()}}
}

/// std::forward_list-like containers: No iterators are invalidated.

void forward_list_push_front(std::forward_list<int> &FL, int n) {
  auto i0 = FL.cbegin(), i1 = FL.cend();

  clang_analyzer_denote(clang_analyzer_container_begin(FL), "$FL.begin()");
  clang_analyzer_denote(clang_analyzer_container_end(FL), "$FL.end()");

  FL.push_front(n);

  clang_analyzer_eval(clang_analyzer_iterator_validity(i0)); //expected-warning{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i1)); //expected-warning{{TRUE}}

  clang_analyzer_express(clang_analyzer_container_begin(FL)); // expected-warning{{$FL.begin() - 1}}
  clang_analyzer_express(clang_analyzer_iterator_position(i0)); // expected-warning{{$FL.begin()}}

  clang_analyzer_express(clang_analyzer_container_end(FL)); // expected-warning{{$FL.end()}}
  clang_analyzer_express(clang_analyzer_iterator_position(i1)); // expected-warning{{$FL.end()}}
}

/// emplace_front()
///
/// - Design decision: extends containers to the <-LEFT<- (i.e. the first
///                    position of the container is decremented).
///
/// - Iterator invalidation rules depend the container type.

/// std::list-like containers: No iterators are invalidated.

void list_emplace_front(std::list<int> &L, int n) {
  auto i0 = L.cbegin(), i1 = L.cend();

  clang_analyzer_denote(clang_analyzer_container_begin(L), "$L.begin()");
  clang_analyzer_denote(clang_analyzer_container_end(L), "$L.end()");

  L.emplace_front(n);

  clang_analyzer_eval(clang_analyzer_iterator_validity(i0)); //expected-warning{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i1)); //expected-warning{{TRUE}}

  clang_analyzer_express(clang_analyzer_container_begin(L)); // expected-warning{{$L.begin() - 1}}
  clang_analyzer_express(clang_analyzer_iterator_position(i0)); // expected-warning{{$L.begin()}}

  clang_analyzer_express(clang_analyzer_container_end(L)); // expected-warning{{$L.end()}}
  clang_analyzer_express(clang_analyzer_iterator_position(i1)); // expected-warning{{$L.end()}}
}

/// std::deque-like containers: All iterators, including the past-the-end
///                             iterator, are invalidated.

void deque_emplace_front(std::deque<int> &D, int n) {
  auto i0 = D.cbegin(), i1 = --D.cend(), i2 = D.cend();

  clang_analyzer_denote(clang_analyzer_container_begin(D), "$D.begin()");
  clang_analyzer_denote(clang_analyzer_container_end(D), "$D.end()");

  D.emplace_front(n);

  clang_analyzer_eval(clang_analyzer_iterator_validity(i0)); //expected-warning{{FALSE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i1)); //expected-warning{{FALSE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i2)); //expected-warning{{FALSE}}

  clang_analyzer_express(clang_analyzer_container_begin(D)); // expected-warning{{$D.begin()}} FIXME: Should be $D.begin - 1 (to correctly track the container's size)

  clang_analyzer_express(clang_analyzer_container_end(D)); // expected-warning{{$D.end()}}
}

/// std::forward_list-like containers: No iterators are invalidated.

void forward_list_emplace_front(std::forward_list<int> &FL, int n) {
  auto i0 = FL.cbegin(), i1 = FL.cend();

  clang_analyzer_denote(clang_analyzer_container_begin(FL), "$FL.begin()");
  clang_analyzer_denote(clang_analyzer_container_end(FL), "$FL.end()");

  FL.emplace_front(n);

  clang_analyzer_eval(clang_analyzer_iterator_validity(i0)); //expected-warning{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i1)); //expected-warning{{TRUE}}

  clang_analyzer_express(clang_analyzer_container_begin(FL)); // expected-warning{{$FL.begin() - 1}}
  clang_analyzer_express(clang_analyzer_iterator_position(i0)); // expected-warning{{$FL.begin()}}

  clang_analyzer_express(clang_analyzer_container_end(FL)); // expected-warning{{$FL.end()}}
  clang_analyzer_express(clang_analyzer_iterator_position(i1)); // expected-warning{{$FL.end()}}
}

/// pop_front()
///
/// - Design decision: shrinks containers to the ->RIGHT-> (i.e. the first
///   position of the container is incremented).
///
/// - Iterator invalidation rules depend the container type.

/// std::list-like containers: Iterators to the first element are invalidated.

void list_pop_front(std::list<int> &L, int n) {
  auto i0 = L.cbegin(), i1 = ++L.cbegin(), i2 = L.cend();

  clang_analyzer_denote(clang_analyzer_container_begin(L), "$L.begin()");
  clang_analyzer_denote(clang_analyzer_container_end(L), "$L.end()");

  L.pop_front();

  clang_analyzer_eval(clang_analyzer_iterator_validity(i0)); //expected-warning{{FALSE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i1)); //expected-warning{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i2)); //expected-warning{{TRUE}}

  clang_analyzer_express(clang_analyzer_container_begin(L)); // expected-warning{{$L.begin() + 1}}
  clang_analyzer_express(clang_analyzer_iterator_position(i1)); // expected-warning{{$L.begin() + 1}}

  clang_analyzer_express(clang_analyzer_container_end(L)); // expected-warning{{$L.end()}}
  clang_analyzer_express(clang_analyzer_iterator_position(i2)); // expected-warning{{$L.end()}}
}

/// std::deque-like containers: Iterators to the first element are invalidated.
///                             Other iterators are not affected.

void deque_pop_front(std::deque<int> &D, int n) {
  auto i0 = D.cbegin(), i1 = ++D.cbegin(), i2 = D.cend();

  clang_analyzer_denote(clang_analyzer_container_begin(D), "$D.begin()");
  clang_analyzer_denote(clang_analyzer_container_end(D), "$D.end()");

  D.pop_front();

  clang_analyzer_eval(clang_analyzer_iterator_validity(i0)); //expected-warning{{FALSE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i1)); //expected-warning{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i2)); //expected-warning{{TRUE}}

  clang_analyzer_express(clang_analyzer_container_begin(D)); // expected-warning{{$D.begin() + 1}}
  clang_analyzer_express(clang_analyzer_iterator_position(i1)); // expected-warning{{$D.begin() + 1}}

  clang_analyzer_express(clang_analyzer_container_end(D)); // expected-warning{{$D.end()}}
  clang_analyzer_express(clang_analyzer_iterator_position(i2)); // expected-warning{{$D.end()}}
}

/// std::forward_list-like containers: Iterators to the first element are
///                                    invalidated.

void forward_list_pop_front(std::list<int> &FL, int n) {
  auto i0 = FL.cbegin(), i1 = ++FL.cbegin(), i2 = FL.cend();

  clang_analyzer_denote(clang_analyzer_container_begin(FL), "$FL.begin()");
  clang_analyzer_denote(clang_analyzer_container_end(FL), "$FL.end()");

  FL.pop_front();

  clang_analyzer_eval(clang_analyzer_iterator_validity(i0)); //expected-warning{{FALSE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i1)); //expected-warning{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i2)); //expected-warning{{TRUE}}

  clang_analyzer_express(clang_analyzer_container_begin(FL)); // expected-warning{{$FL.begin() + 1}}
  clang_analyzer_express(clang_analyzer_iterator_position(i1)); // expected-warning{{$FL.begin() + 1}}

  clang_analyzer_express(clang_analyzer_container_end(FL)); // expected-warning{{$FL.end()}}
  clang_analyzer_express(clang_analyzer_iterator_position(i2)); // expected-warning{{$FL.end()}}
}

/// insert()
///
/// - Design decision: shifts positions to the <-LEFT<- (i.e. all iterator
///                    ahead of the insertion point are decremented; if the
///                    relation between the insertion point and the first
///                    position of the container is known, the first position
///                    of the container is also decremented).
///
/// - Iterator invalidation rules depend the container type.

/// std::list-like containers: No iterators are invalidated.

void list_insert_begin(std::list<int> &L, int n) {
  auto i0 = L.cbegin(), i1 = L.cend();

  clang_analyzer_denote(clang_analyzer_container_begin(L), "$L.begin()");
  clang_analyzer_denote(clang_analyzer_container_end(L), "$L.end()");

  auto i2 = L.insert(i0, n);

  clang_analyzer_eval(clang_analyzer_iterator_validity(i0)); //expected-warning{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i1)); //expected-warning{{TRUE}}

  clang_analyzer_express(clang_analyzer_container_begin(L)); // expected-warning{{$L.begin()}} FIXME: Should be $L.begin() - 1
  clang_analyzer_express(clang_analyzer_iterator_position(i0)); // expected-warning{{$L.begin()}}
  // clang_analyzer_express(clang_analyzer_iterator_position(i2)); FIXME: expect warning $L.begin() - 1

  clang_analyzer_express(clang_analyzer_container_end(L)); // expected-warning{{$L.end()}}
  clang_analyzer_express(clang_analyzer_iterator_position(i1)); // expected-warning{{$L.end()}}
}

void list_insert_behind_begin(std::list<int> &L, int n) {
  auto i0 = L.cbegin(), i1 = ++L.cbegin(), i2 = L.cend();

  clang_analyzer_denote(clang_analyzer_container_begin(L), "$L.begin()");
  clang_analyzer_denote(clang_analyzer_container_end(L), "$L.end()");

  auto i3 = L.insert(i1, n);

  clang_analyzer_eval(clang_analyzer_iterator_validity(i0)); //expected-warning{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i1)); //expected-warning{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i2)); //expected-warning{{TRUE}}

  clang_analyzer_express(clang_analyzer_container_begin(L)); // expected-warning{{$L.begin()}} FIXME: Should be $L.begin() - 1
  clang_analyzer_express(clang_analyzer_iterator_position(i0)); // expected-warning{{$L.begin()}} FIXME: Should be $L.begin() - 1
  clang_analyzer_express(clang_analyzer_iterator_position(i1)); // expected-warning{{$L.begin() + 1}}
  // clang_analyzer_express(clang_analyzer_iterator_position(i3)); FIXME: expect warning $L.begin()

  clang_analyzer_express(clang_analyzer_container_end(L)); // expected-warning{{$L.end()}}
  clang_analyzer_express(clang_analyzer_iterator_position(i2)); // expected-warning{{$L.end()}}
}

template <typename Iter> Iter return_any_iterator(const Iter &It);

void list_insert_unknown(std::list<int> &L, int n) {
  auto i0 = L.cbegin(), i1 = return_any_iterator(L.cbegin()), i2 = L.cend();

  clang_analyzer_denote(clang_analyzer_container_begin(L), "$L.begin()");
  clang_analyzer_denote(clang_analyzer_container_end(L), "$L.end()");
  clang_analyzer_denote(clang_analyzer_iterator_position(i1), "$i1");

  auto i3 = L.insert(i1, n);

  clang_analyzer_eval(clang_analyzer_iterator_validity(i0)); //expected-warning{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i1)); //expected-warning{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i2)); //expected-warning{{TRUE}}

  clang_analyzer_express(clang_analyzer_container_begin(L)); // expected-warning{{$L.begin()}}
  clang_analyzer_express(clang_analyzer_iterator_position(i0)); // expected-warning{{$L.begin()}}

  clang_analyzer_express(clang_analyzer_iterator_position(i1)); // expected-warning{{$i1}}
  // clang_analyzer_express(clang_analyzer_iterator_position(i3)); FIXME: expect warning $i - 1

  clang_analyzer_express(clang_analyzer_container_end(L)); // expected-warning{{$L.end()}}
  clang_analyzer_express(clang_analyzer_iterator_position(i2)); // expected-warning{{$L.end()}}
}

void list_insert_ahead_of_end(std::list<int> &L, int n) {
  auto i0 = L.cbegin(), i1 = --L.cend(), i2 = L.cend();

  clang_analyzer_denote(clang_analyzer_container_begin(L), "$L.begin()");
  clang_analyzer_denote(clang_analyzer_container_end(L), "$L.end()");

  auto i3 = L.insert(i1, n);

  clang_analyzer_eval(clang_analyzer_iterator_validity(i0)); //expected-warning{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i1)); //expected-warning{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i2)); //expected-warning{{TRUE}}

  clang_analyzer_express(clang_analyzer_container_begin(L)); // expected-warning{{$L.begin()}}
  clang_analyzer_express(clang_analyzer_iterator_position(i0)); // expected-warning{{$L.begin()}}

  clang_analyzer_express(clang_analyzer_container_end(L)); // expected-warning{{$L.end()}}
  clang_analyzer_express(clang_analyzer_iterator_position(i1)); // expected-warning{{$L.end() - 1}}
  clang_analyzer_express(clang_analyzer_iterator_position(i2)); // expected-warning{{$L.end()}}
  // clang_analyzer_express(clang_analyzer_iterator_position(i3)); FIXME: expect warning $L.end() - 2
}

void list_insert_end(std::list<int> &L, int n) {
  auto i0 = L.cbegin(), i1 = --L.cend(), i2 = L.cend();

  clang_analyzer_denote(clang_analyzer_container_begin(L), "$L.begin()");
  clang_analyzer_denote(clang_analyzer_container_end(L), "$L.end()");

  auto i3 = L.insert(i2, n);

  clang_analyzer_eval(clang_analyzer_iterator_validity(i0)); //expected-warning{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i1)); //expected-warning{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i2)); //expected-warning{{TRUE}}

  clang_analyzer_express(clang_analyzer_container_begin(L)); // expected-warning{{$L.begin()}}
  clang_analyzer_express(clang_analyzer_iterator_position(i0)); // expected-warning{{$L.begin()}}

  clang_analyzer_express(clang_analyzer_container_end(L)); // expected-warning{{$L.end()}}
  clang_analyzer_express(clang_analyzer_iterator_position(i1)); // expected-warning{{$L.end() - 1}} FIXME: should be $L.end() - 2
  clang_analyzer_express(clang_analyzer_iterator_position(i2)); // expected-warning{{$L.end()}}
  // clang_analyzer_express(clang_analyzer_iterator_position(i3)); FIXME: expect warning $L.end() - 1
}

/// std::vector-like containers: Only the iterators before the insertion point
///                              remain valid. The past-the-end iterator is also
///                              invalidated.

void vector_insert_begin(std::vector<int> &V, int n) {
  auto i0 = V.cbegin(), i1 = V.cend();

  clang_analyzer_denote(clang_analyzer_container_begin(V), "$V.begin()");
  clang_analyzer_denote(clang_analyzer_container_end(V), "$V.end()");

  auto i2 = V.insert(i0, n);

  clang_analyzer_eval(clang_analyzer_iterator_validity(i0)); //expected-warning{{FALSE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i1)); //expected-warning{{FALSE}}

  clang_analyzer_express(clang_analyzer_container_begin(V)); // expected-warning{{$V.begin()}} FIXME: Should be $V.begin() - 1
  // clang_analyzer_express(clang_analyzer_iterator_position(i2)); FIXME: expect warning $V.begin() - 1

  // clang_analyzer_express(clang_analyzer_container_end(V)); // FIXME: expect warning $V.end()
}

void vector_insert_behind_begin(std::vector<int> &V, int n) {
  auto i0 = V.cbegin(), i1 = ++V.cbegin(), i2 = V.cend();

  clang_analyzer_denote(clang_analyzer_container_begin(V), "$V.begin()");
  clang_analyzer_denote(clang_analyzer_container_end(V), "$V.end()");

  auto i3 = V.insert(i1, n);

  clang_analyzer_eval(clang_analyzer_iterator_validity(i0)); //expected-warning{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i1)); //expected-warning{{FALSE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i2)); //expected-warning{{FALSE}}

  clang_analyzer_express(clang_analyzer_container_begin(V)); // expected-warning{{$V.begin()}} FIXME: Should be $V.begin() - 1
  clang_analyzer_express(clang_analyzer_iterator_position(i0)); // expected-warning{{$V.begin()}} FIXME: Should be $V.begin() - 1
  // clang_analyzer_express(clang_analyzer_iterator_position(i3)); // FIXME: expect -warning $V.begin()

  // clang_analyzer_express(clang_analyzer_container_end(V)); // FIXME: expect warning $V.end()
}

void vector_insert_unknown(std::vector<int> &V, int n) {
  auto i0 = V.cbegin(), i1 = return_any_iterator(V.cbegin()), i2 = V.cend();

  clang_analyzer_denote(clang_analyzer_container_begin(V), "$V.begin()");
  clang_analyzer_denote(clang_analyzer_container_end(V), "$V.end()");
  clang_analyzer_denote(clang_analyzer_iterator_position(i1), "$i1");

  auto i3 = V.insert(i1, n);

  clang_analyzer_eval(clang_analyzer_iterator_validity(i0)); //expected-warning{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i1)); //expected-warning{{FALSE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i2)); //expected-warning{{FALSE}}

  clang_analyzer_express(clang_analyzer_container_begin(V)); // expected-warning{{$V.begin()}}
  clang_analyzer_express(clang_analyzer_iterator_position(i0)); // expected-warning{{$V.begin()}}

  // clang_analyzer_express(clang_analyzer_iterator_position(i3)); FIXME: expecte warning $i1 - 1

  // clang_analyzer_express(clang_analyzer_container_end(V)); FIXME expect warning $V.end()
}

void vector_insert_ahead_of_end(std::vector<int> &V, int n) {
  auto i0 = V.cbegin(), i1 = --V.cend(), i2 = V.cend();

  clang_analyzer_denote(clang_analyzer_container_begin(V), "$V.begin()");
  clang_analyzer_denote(clang_analyzer_container_end(V), "$V.end()");

  auto i3 = V.insert(i1, n);

  clang_analyzer_eval(clang_analyzer_iterator_validity(i0)); //expected-warning{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i1)); //expected-warning{{FALSE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i2)); //expected-warning{{FALSE}}

  clang_analyzer_express(clang_analyzer_container_begin(V)); // expected-warning{{$V.begin()}}
  clang_analyzer_express(clang_analyzer_iterator_position(i0)); // expected-warning{{$V.begin()}}

  // clang_analyzer_express(clang_analyzer_container_end(V)); FIXME: expect warning $V.end() + 1
  // clang_analyzer_express(clang_analyzer_iterator_position(i3)); FIXME: expect warning $V.end() - 2
}

void vector_insert_end(std::vector<int> &V, int n) {
  auto i0 = V.cbegin(), i1 = --V.cend(), i2 = V.cend();

  clang_analyzer_denote(clang_analyzer_container_begin(V), "$V.begin()");
  clang_analyzer_denote(clang_analyzer_container_end(V), "$V.end()");

  auto i3 = V.insert(i2, n);

  clang_analyzer_eval(clang_analyzer_iterator_validity(i0)); //expected-warning{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i1)); //expected-warning{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i2)); //expected-warning{{FALSE}}

  clang_analyzer_express(clang_analyzer_container_begin(V)); // expected-warning{{$V.begin()}}
  clang_analyzer_express(clang_analyzer_iterator_position(i0)); // expected-warning{{$V.begin()}}

  // clang_analyzer_express(clang_analyzer_container_end(V)); FIXME: expect warning $V.end()
  clang_analyzer_express(clang_analyzer_iterator_position(i1)); // expected-warning{{$V.end() - 1}} FIXME: Should be $V.end() - 2
  // clang_analyzer_express(clang_analyzer_iterator_position(i3)); FIXME: expect warning $V.end() - 1
}

/// std::deque-like containers: All iterators, including the past-the-end
///                             iterator, are invalidated.

void deque_insert_begin(std::deque<int> &D, int n) {
  auto i0 = D.cbegin(), i1 = D.cend();

  clang_analyzer_denote(clang_analyzer_container_begin(D), "$D.begin()");
  clang_analyzer_denote(clang_analyzer_container_end(D), "$D.end()");

  auto i2 = D.insert(i0, n);

  clang_analyzer_eval(clang_analyzer_iterator_validity(i0)); //expected-warning{{FALSE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i1)); //expected-warning{{FALSE}}

  clang_analyzer_express(clang_analyzer_container_begin(D)); // expected-warning{{$D.begin()}} FIXME: Should be $D.begin() - 1
  // clang_analyzer_express(clang_analyzer_iterator_position(i2)); FIXME: expect warning $D.begin() - 1

  // clang_analyzer_express(clang_analyzer_container_end(D)); FIXME: expect warning $D.end()
}

void deque_insert_behind_begin(std::deque<int> &D, int n) {
  auto i0 = D.cbegin(), i1 = ++D.cbegin(), i2 = D.cend();

  clang_analyzer_denote(clang_analyzer_container_begin(D), "$D.begin()");
  clang_analyzer_denote(clang_analyzer_container_end(D), "$D.end()");

  auto i3 = D.insert(i1, n);

  clang_analyzer_eval(clang_analyzer_iterator_validity(i0)); //expected-warning{{FALSE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i1)); //expected-warning{{FALSE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i2)); //expected-warning{{FALSE}}

  clang_analyzer_express(clang_analyzer_container_begin(D)); // expected-warning{{$D.begin()}} FIXME: Should be $D.begin - 1
  // clang_analyzer_express(clang_analyzer_iterator_position(i3)); FIXME: expect warning $D.begin() - 1

  // clang_analyzer_express(clang_analyzer_container_end(D)); FIXME: expect warning $D.end()
}

void deque_insert_unknown(std::deque<int> &D, int n) {
  auto i0 = D.cbegin(), i1 = return_any_iterator(D.cbegin()), i2 = D.cend();

  clang_analyzer_denote(clang_analyzer_container_begin(D), "$D.begin()");
  clang_analyzer_denote(clang_analyzer_container_end(D), "$D.end()");
  clang_analyzer_denote(clang_analyzer_iterator_position(i1), "$i1");

  auto i3 = D.insert(i1, n);

  clang_analyzer_eval(clang_analyzer_iterator_validity(i0)); //expected-warning{{FALSE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i1)); //expected-warning{{FALSE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i2)); //expected-warning{{FALSE}}

  clang_analyzer_express(clang_analyzer_container_begin(D)); // expected-warning{{$D.begin()}}

  // clang_analyzer_express(clang_analyzer_iterator_position(i3)); FIXME: expect warning $i1 - 1

  // clang_analyzer_express(clang_analyzer_container_end(D)); FIXME: expect warning $D.end()
}

void deque_insert_ahead_of_end(std::deque<int> &D, int n) {
  auto i0 = D.cbegin(), i1 = --D.cend(), i2 = D.cend();

  clang_analyzer_denote(clang_analyzer_container_begin(D), "$D.begin()");
  clang_analyzer_denote(clang_analyzer_container_end(D), "$D.end()");

  auto i3 = D.insert(i1, n);

  clang_analyzer_eval(clang_analyzer_iterator_validity(i0)); //expected-warning{{FALSE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i1)); //expected-warning{{FALSE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i2)); //expected-warning{{FALSE}}

  clang_analyzer_express(clang_analyzer_container_begin(D)); // expected-warning{{$D.begin()}}

  // clang_analyzer_express(clang_analyzer_container_end(D)); FIXME: expect warning $D.end() + 1
  // clang_analyzer_express(clang_analyzer_iterator_position(i3)); FIXME: expect warning $D.end() - 2
}

void deque_insert_end(std::deque<int> &D, int n) {
  auto i0 = D.cbegin(), i1 = --D.cend(), i2 = D.cend();

  clang_analyzer_denote(clang_analyzer_container_begin(D), "$D.begin()");
  clang_analyzer_denote(clang_analyzer_container_end(D), "$D.end()");

  auto i3 = D.insert(i2, n);

  clang_analyzer_eval(clang_analyzer_iterator_validity(i0)); //expected-warning{{FALSE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i1)); //expected-warning{{FALSE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i2)); //expected-warning{{FALSE}}

  clang_analyzer_express(clang_analyzer_container_begin(D)); // expected-warning{{$D.begin()}}

  // clang_analyzer_express(clang_analyzer_container_end(D)); FIXME: expect warning $D.end()
  // clang_analyzer_express(clang_analyzer_iterator_position(i3)); FIXME: expect warning $D.end() - 1
}

/// insert_after()   [std::forward_list-like containers]
///
/// - Design decision: shifts positions to the ->RIGHT-> (i.e. all iterator
///                    ahead of the insertion point are incremented; if the
///                    relation between the insertion point and the past-the-end
///                    position of the container is known, the first position of
///                    the container is also incremented).
///
/// - No iterators are invalidated.

void forward_list_insert_after_begin(std::forward_list<int> &FL, int n) {
  auto i0 = FL.cbegin(), i1 = FL.cend();

  clang_analyzer_denote(clang_analyzer_container_begin(FL), "$FL.begin()");
  clang_analyzer_denote(clang_analyzer_container_end(FL), "$FL.end()");

  auto i2 = FL.insert_after(i0, n);

  clang_analyzer_eval(clang_analyzer_iterator_validity(i0)); //expected-warning{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i1)); //expected-warning{{TRUE}}

  clang_analyzer_express(clang_analyzer_container_begin(FL)); // expected-warning{{$FL.begin()}}
  clang_analyzer_express(clang_analyzer_iterator_position(i0)); // expected-warning{{$FL.begin()}}
  // clang_analyzer_express(clang_analyzer_iterator_position(i2)); FIXME: expect warning $FL.begin() + 1

  clang_analyzer_express(clang_analyzer_container_end(FL)); // expected-warning{{$FL.end()}}
  clang_analyzer_express(clang_analyzer_iterator_position(i1)); // expected-warning{{$FL.end()}}
}

void forward_list_insert_after_behind_begin(std::forward_list<int> &FL, int n) {
  auto i0 = FL.cbegin(), i1 = ++FL.cbegin(), i2 = FL.cend();

  clang_analyzer_denote(clang_analyzer_container_begin(FL), "$FL.begin()");
  clang_analyzer_denote(clang_analyzer_container_end(FL), "$FL.end()");

  auto i3 = FL.insert_after(i1, n);

  clang_analyzer_eval(clang_analyzer_iterator_validity(i0)); //expected-warning{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i1)); //expected-warning{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i2)); //expected-warning{{TRUE}}

  clang_analyzer_express(clang_analyzer_container_begin(FL)); // expected-warning{{$FL.begin()}}
  clang_analyzer_express(clang_analyzer_iterator_position(i0)); // expected-warning{{$FL.begin()}}
  clang_analyzer_express(clang_analyzer_iterator_position(i1)); // expected-warning{{$FL.begin() + 1}}
  // clang_analyzer_express(clang_analyzer_iterator_position(i3)); FIXME: expect warning $FL.begin() + 2

  clang_analyzer_express(clang_analyzer_container_end(FL)); // expected-warning{{$FL.end()}}
  clang_analyzer_express(clang_analyzer_iterator_position(i2)); // expected-warning{{$FL.end()}}
}

void forward_list_insert_after_unknown(std::forward_list<int> &FL, int n) {
  auto i0 = FL.cbegin(), i1 = return_any_iterator(FL.cbegin()), i2 = FL.cend();

  clang_analyzer_denote(clang_analyzer_container_begin(FL), "$FL.begin()");
  clang_analyzer_denote(clang_analyzer_container_end(FL), "$FL.end()");
  clang_analyzer_denote(clang_analyzer_iterator_position(i1), "$i1");

  auto i3 = FL.insert_after(i1, n);

  clang_analyzer_eval(clang_analyzer_iterator_validity(i0)); //expected-warning{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i1)); //expected-warning{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i2)); //expected-warning{{TRUE}}

  clang_analyzer_express(clang_analyzer_container_begin(FL)); // expected-warning{{$FL.begin()}}
  clang_analyzer_express(clang_analyzer_iterator_position(i0)); // expected-warning{{$FL.begin()}}

  clang_analyzer_express(clang_analyzer_iterator_position(i1)); // expected-warning{{$i1}}
  // clang_analyzer_express(clang_analyzer_iterator_position(i3)); FIXME: expect warning $i1 + 1

  clang_analyzer_express(clang_analyzer_container_end(FL)); // expected-warning{{$FL.end()}}
  clang_analyzer_express(clang_analyzer_iterator_position(i2)); // expected-warning{{$FL.end()}}
}

/// emplace()
///
/// - Design decision: shifts positions to the <-LEFT<- (i.e. all iterator
///                    ahead of the emplacement point are decremented; if the
///                    relation between the emplacement point and the first
///                    position of the container is known, the first position
///                    of the container is also decremented).
///
/// - Iterator invalidation rules depend the container type.

/// std::list-like containers: No iterators are invalidated.

void list_emplace_begin(std::list<int> &L, int n) {
  auto i0 = L.cbegin(), i1 = L.cend();

  clang_analyzer_denote(clang_analyzer_container_begin(L), "$L.begin()");
  clang_analyzer_denote(clang_analyzer_container_end(L), "$L.end()");

  auto i2 = L.emplace(i0, n);

  clang_analyzer_eval(clang_analyzer_iterator_validity(i0)); //expected-warning{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i1)); //expected-warning{{TRUE}}

  clang_analyzer_express(clang_analyzer_container_begin(L)); // expected-warning{{$L.begin()}} FIXME: Should be $L.begin() - 1
  clang_analyzer_express(clang_analyzer_iterator_position(i0)); // expected-warning{{$L.begin()}}
  // clang_analyzer_express(clang_analyzer_iterator_position(i2)); FIXME: expect warning $L.begin() - 1

  clang_analyzer_express(clang_analyzer_container_end(L)); // expected-warning{{$L.end()}}
  clang_analyzer_express(clang_analyzer_iterator_position(i1)); // expected-warning{{$L.end()}}
}

void list_emplace_behind_begin(std::list<int> &L, int n) {
  auto i0 = L.cbegin(), i1 = ++L.cbegin(), i2 = L.cend();

  clang_analyzer_denote(clang_analyzer_container_begin(L), "$L.begin()");
  clang_analyzer_denote(clang_analyzer_container_end(L), "$L.end()");

  auto i3 = L.emplace(i1, n);

  clang_analyzer_eval(clang_analyzer_iterator_validity(i0)); //expected-warning{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i1)); //expected-warning{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i2)); //expected-warning{{TRUE}}

  clang_analyzer_express(clang_analyzer_container_begin(L)); // expected-warning{{$L.begin()}} FIXME: Should be $L.begin() - 1
  clang_analyzer_express(clang_analyzer_iterator_position(i0)); // expected-warning{{$L.begin()}} FIXME: Should be $L.begin() - 1
  clang_analyzer_express(clang_analyzer_iterator_position(i1)); // expected-warning{{$L.begin() + 1}}
  // clang_analyzer_express(clang_analyzer_iterator_position(i3)); FIXME: expect warning $L.begin()

  clang_analyzer_express(clang_analyzer_container_end(L)); // expected-warning{{$L.end()}}
  clang_analyzer_express(clang_analyzer_iterator_position(i2)); // expected-warning{{$L.end()}}
}

template <typename Iter> Iter return_any_iterator(const Iter &It);

void list_emplace_unknown(std::list<int> &L, int n) {
  auto i0 = L.cbegin(), i1 = return_any_iterator(L.cbegin()), i2 = L.cend();

  clang_analyzer_denote(clang_analyzer_container_begin(L), "$L.begin()");
  clang_analyzer_denote(clang_analyzer_container_end(L), "$L.end()");
  clang_analyzer_denote(clang_analyzer_iterator_position(i1), "$i1");

  auto i3 = L.emplace(i1, n);

  clang_analyzer_eval(clang_analyzer_iterator_validity(i0)); //expected-warning{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i1)); //expected-warning{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i2)); //expected-warning{{TRUE}}

  clang_analyzer_express(clang_analyzer_container_begin(L)); // expected-warning{{$L.begin()}}
  clang_analyzer_express(clang_analyzer_iterator_position(i0)); // expected-warning{{$L.begin()}}

  clang_analyzer_express(clang_analyzer_iterator_position(i1)); // expected-warning{{$i1}}
  // clang_analyzer_express(clang_analyzer_iterator_position(i3)); FIXME: expect warning $i - 1

  clang_analyzer_express(clang_analyzer_container_end(L)); // expected-warning{{$L.end()}}
  clang_analyzer_express(clang_analyzer_iterator_position(i2)); // expected-warning{{$L.end()}}
}

void list_emplace_ahead_of_end(std::list<int> &L, int n) {
  auto i0 = L.cbegin(), i1 = --L.cend(), i2 = L.cend();

  clang_analyzer_denote(clang_analyzer_container_begin(L), "$L.begin()");
  clang_analyzer_denote(clang_analyzer_container_end(L), "$L.end()");

  auto i3 = L.emplace(i1, n);

  clang_analyzer_eval(clang_analyzer_iterator_validity(i0)); //expected-warning{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i1)); //expected-warning{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i2)); //expected-warning{{TRUE}}

  clang_analyzer_express(clang_analyzer_container_begin(L)); // expected-warning{{$L.begin()}}
  clang_analyzer_express(clang_analyzer_iterator_position(i0)); // expected-warning{{$L.begin()}}

  clang_analyzer_express(clang_analyzer_container_end(L)); // expected-warning{{$L.end()}}
  clang_analyzer_express(clang_analyzer_iterator_position(i1)); // expected-warning{{$L.end() - 1}}
  clang_analyzer_express(clang_analyzer_iterator_position(i2)); // expected-warning{{$L.end()}}
  // clang_analyzer_express(clang_analyzer_iterator_position(i3)); FIXME: expect warning $L.end() - 2
}

void list_emplace_end(std::list<int> &L, int n) {
  auto i0 = L.cbegin(), i1 = --L.cend(), i2 = L.cend();

  clang_analyzer_denote(clang_analyzer_container_begin(L), "$L.begin()");
  clang_analyzer_denote(clang_analyzer_container_end(L), "$L.end()");

  auto i3 = L.emplace(i2, n);

  clang_analyzer_eval(clang_analyzer_iterator_validity(i0)); //expected-warning{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i1)); //expected-warning{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i2)); //expected-warning{{TRUE}}

  clang_analyzer_express(clang_analyzer_container_begin(L)); // expected-warning{{$L.begin()}}
  clang_analyzer_express(clang_analyzer_iterator_position(i0)); // expected-warning{{$L.begin()}}

  clang_analyzer_express(clang_analyzer_container_end(L)); // expected-warning{{$L.end()}}
  clang_analyzer_express(clang_analyzer_iterator_position(i1)); // expected-warning{{$L.end() - 1}} FIXME: should be $L.end() - 2
  clang_analyzer_express(clang_analyzer_iterator_position(i2)); // expected-warning{{$L.end()}}
  // clang_analyzer_express(clang_analyzer_iterator_position(i3)); FIXME: expect warning $L.end() - 1
}

/// std::vector-like containers: Only the iterators before the emplacement point
///                              remain valid. The past-the-end iterator is also
///                              invalidated.

void vector_emplace_begin(std::vector<int> &V, int n) {
  auto i0 = V.cbegin(), i1 = V.cend();

  clang_analyzer_denote(clang_analyzer_container_begin(V), "$V.begin()");
  clang_analyzer_denote(clang_analyzer_container_end(V), "$V.end()");

  auto i2 = V.emplace(i0, n);

  clang_analyzer_eval(clang_analyzer_iterator_validity(i0)); //expected-warning{{FALSE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i1)); //expected-warning{{FALSE}}

  clang_analyzer_express(clang_analyzer_container_begin(V)); // expected-warning{{$V.begin()}} FIXME: Should be $V.begin() - 1
  // clang_analyzer_express(clang_analyzer_iterator_position(i2)); FIXME: expect warning $V.begin() - 1

  // clang_analyzer_express(clang_analyzer_container_end(V)); // FIXME: expect warning $V.end()
}

void vector_emplace_behind_begin(std::vector<int> &V, int n) {
  auto i0 = V.cbegin(), i1 = ++V.cbegin(), i2 = V.cend();

  clang_analyzer_denote(clang_analyzer_container_begin(V), "$V.begin()");
  clang_analyzer_denote(clang_analyzer_container_end(V), "$V.end()");

  auto i3 = V.emplace(i1, n);

  clang_analyzer_eval(clang_analyzer_iterator_validity(i0)); //expected-warning{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i1)); //expected-warning{{FALSE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i2)); //expected-warning{{FALSE}}

  clang_analyzer_express(clang_analyzer_container_begin(V)); // expected-warning{{$V.begin()}} FIXME: Should be $V.begin() - 1
  clang_analyzer_express(clang_analyzer_iterator_position(i0)); // expected-warning{{$V.begin()}} FIXME: Should be $V.begin() - 1
  // clang_analyzer_express(clang_analyzer_iterator_position(i3)); // FIXME: expect -warning $V.begin()

  // clang_analyzer_express(clang_analyzer_container_end(V)); // FIXME: expect warning $V.end()
}

void vector_emplace_unknown(std::vector<int> &V, int n) {
  auto i0 = V.cbegin(), i1 = return_any_iterator(V.cbegin()), i2 = V.cend();

  clang_analyzer_denote(clang_analyzer_container_begin(V), "$V.begin()");
  clang_analyzer_denote(clang_analyzer_container_end(V), "$V.end()");
  clang_analyzer_denote(clang_analyzer_iterator_position(i1), "$i1");

  auto i3 = V.emplace(i1, n);

  clang_analyzer_eval(clang_analyzer_iterator_validity(i0)); //expected-warning{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i1)); //expected-warning{{FALSE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i2)); //expected-warning{{FALSE}}

  clang_analyzer_express(clang_analyzer_container_begin(V)); // expected-warning{{$V.begin()}}
  clang_analyzer_express(clang_analyzer_iterator_position(i0)); // expected-warning{{$V.begin()}}

  // clang_analyzer_express(clang_analyzer_iterator_position(i3)); FIXME: expecte warning $i1 - 1

  // clang_analyzer_express(clang_analyzer_container_end(V)); FIXME expect warning $V.end()
}

void vector_emplace_ahead_of_end(std::vector<int> &V, int n) {
  auto i0 = V.cbegin(), i1 = --V.cend(), i2 = V.cend();

  clang_analyzer_denote(clang_analyzer_container_begin(V), "$V.begin()");
  clang_analyzer_denote(clang_analyzer_container_end(V), "$V.end()");

  auto i3 = V.emplace(i1, n);

  clang_analyzer_eval(clang_analyzer_iterator_validity(i0)); //expected-warning{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i1)); //expected-warning{{FALSE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i2)); //expected-warning{{FALSE}}

  clang_analyzer_express(clang_analyzer_container_begin(V)); // expected-warning{{$V.begin()}}
  clang_analyzer_express(clang_analyzer_iterator_position(i0)); // expected-warning{{$V.begin()}}

  // clang_analyzer_express(clang_analyzer_container_end(V)); FIXME: expect warning $V.end() + 1
  // clang_analyzer_express(clang_analyzer_iterator_position(i3)); FIXME: expect warning $V.end() - 2
}

void vector_emplace_end(std::vector<int> &V, int n) {
  auto i0 = V.cbegin(), i1 = --V.cend(), i2 = V.cend();

  clang_analyzer_denote(clang_analyzer_container_begin(V), "$V.begin()");
  clang_analyzer_denote(clang_analyzer_container_end(V), "$V.end()");

  auto i3 = V.emplace(i2, n);

  clang_analyzer_eval(clang_analyzer_iterator_validity(i0)); //expected-warning{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i1)); //expected-warning{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i2)); //expected-warning{{FALSE}}

  clang_analyzer_express(clang_analyzer_container_begin(V)); // expected-warning{{$V.begin()}}
  clang_analyzer_express(clang_analyzer_iterator_position(i0)); // expected-warning{{$V.begin()}}

  // clang_analyzer_express(clang_analyzer_container_end(V)); FIXME: expect warning $V.end()
  clang_analyzer_express(clang_analyzer_iterator_position(i1)); // expected-warning{{$V.end() - 1}} FIXME: Should be $V.end() - 2
  // clang_analyzer_express(clang_analyzer_iterator_position(i3)); FIXME: expect warning $V.end() - 1
}

/// std::deque-like containers: All iterators, including the past-the-end
///                             iterator, are invalidated.

void deque_emplace_begin(std::deque<int> &D, int n) {
  auto i0 = D.cbegin(), i1 = D.cend();

  clang_analyzer_denote(clang_analyzer_container_begin(D), "$D.begin()");
  clang_analyzer_denote(clang_analyzer_container_end(D), "$D.end()");

  auto i2 = D.emplace(i0, n);

  clang_analyzer_eval(clang_analyzer_iterator_validity(i0)); //expected-warning{{FALSE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i1)); //expected-warning{{FALSE}}

  clang_analyzer_express(clang_analyzer_container_begin(D)); // expected-warning{{$D.begin()}} FIXME: Should be $D.begin() - 1
  // clang_analyzer_express(clang_analyzer_iterator_position(i2)); FIXME: expect warning $D.begin() - 1

  // clang_analyzer_express(clang_analyzer_container_end(D)); FIXME: expect warning $D.end()
}

void deque_emplace_behind_begin(std::deque<int> &D, int n) {
  auto i0 = D.cbegin(), i1 = ++D.cbegin(), i2 = D.cend();

  clang_analyzer_denote(clang_analyzer_container_begin(D), "$D.begin()");
  clang_analyzer_denote(clang_analyzer_container_end(D), "$D.end()");

  auto i3 = D.emplace(i1, n);

  clang_analyzer_eval(clang_analyzer_iterator_validity(i0)); //expected-warning{{FALSE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i1)); //expected-warning{{FALSE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i2)); //expected-warning{{FALSE}}

  clang_analyzer_express(clang_analyzer_container_begin(D)); // expected-warning{{$D.begin()}} FIXME: Should be $D.begin - 1
  // clang_analyzer_express(clang_analyzer_iterator_position(i3)); FIXME: expect warning $D.begin() - 1

  // clang_analyzer_express(clang_analyzer_container_end(D)); FIXME: expect warning $D.end()
}

void deque_emplace_unknown(std::deque<int> &D, int n) {
  auto i0 = D.cbegin(), i1 = return_any_iterator(D.cbegin()), i2 = D.cend();

  clang_analyzer_denote(clang_analyzer_container_begin(D), "$D.begin()");
  clang_analyzer_denote(clang_analyzer_container_end(D), "$D.end()");
  clang_analyzer_denote(clang_analyzer_iterator_position(i1), "$i1");

  auto i3 = D.emplace(i1, n);

  clang_analyzer_eval(clang_analyzer_iterator_validity(i0)); //expected-warning{{FALSE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i1)); //expected-warning{{FALSE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i2)); //expected-warning{{FALSE}}

  clang_analyzer_express(clang_analyzer_container_begin(D)); // expected-warning{{$D.begin()}}

  // clang_analyzer_express(clang_analyzer_iterator_position(i3)); FIXME: expect warning $i1 - 1

  // clang_analyzer_express(clang_analyzer_container_end(D)); FIXME: expect warning $D.end()
}

void deque_emplace_ahead_of_end(std::deque<int> &D, int n) {
  auto i0 = D.cbegin(), i1 = --D.cend(), i2 = D.cend();

  clang_analyzer_denote(clang_analyzer_container_begin(D), "$D.begin()");
  clang_analyzer_denote(clang_analyzer_container_end(D), "$D.end()");

  auto i3 = D.emplace(i1, n);

  clang_analyzer_eval(clang_analyzer_iterator_validity(i0)); //expected-warning{{FALSE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i1)); //expected-warning{{FALSE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i2)); //expected-warning{{FALSE}}

  clang_analyzer_express(clang_analyzer_container_begin(D)); // expected-warning{{$D.begin()}}

  // clang_analyzer_express(clang_analyzer_container_end(D)); FIXME: expect warning $D.end() + 1
  // clang_analyzer_express(clang_analyzer_iterator_position(i3)); FIXME: expect warning $D.end() - 2
}

void deque_emplace_end(std::deque<int> &D, int n) {
  auto i0 = D.cbegin(), i1 = --D.cend(), i2 = D.cend();

  clang_analyzer_denote(clang_analyzer_container_begin(D), "$D.begin()");
  clang_analyzer_denote(clang_analyzer_container_end(D), "$D.end()");

  auto i3 = D.emplace(i2, n);

  clang_analyzer_eval(clang_analyzer_iterator_validity(i0)); //expected-warning{{FALSE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i1)); //expected-warning{{FALSE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i2)); //expected-warning{{FALSE}}

  clang_analyzer_express(clang_analyzer_container_begin(D)); // expected-warning{{$D.begin()}}

  // clang_analyzer_express(clang_analyzer_container_end(D)); FIXME: expect warning $D.end()
  // clang_analyzer_express(clang_analyzer_iterator_position(i3)); FIXME: expect warning $D.end() - 1
}

/// emplace_after()   [std::forward_list-like containers]
///
/// - Design decision: shifts positions to the ->RIGHT-> (i.e. all iterator
///                    ahead of the emplacement point are incremented; if the
///                    relation between the emplacement point and the
///                    past-the-end position of the container is known, the
///                    first position of the container is also incremented).
///
/// - No iterators are invalidated.

void forward_list_emplace_after_begin(std::forward_list<int> &FL, int n) {
  auto i0 = FL.cbegin(), i1 = FL.cend();

  clang_analyzer_denote(clang_analyzer_container_begin(FL), "$FL.begin()");
  clang_analyzer_denote(clang_analyzer_container_end(FL), "$FL.end()");

  auto i2 = FL.emplace_after(i0, n);

  clang_analyzer_eval(clang_analyzer_iterator_validity(i0)); //expected-warning{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i1)); //expected-warning{{TRUE}}

  clang_analyzer_express(clang_analyzer_container_begin(FL)); // expected-warning{{$FL.begin()}}
  clang_analyzer_express(clang_analyzer_iterator_position(i0)); // expected-warning{{$FL.begin()}}
  // clang_analyzer_express(clang_analyzer_iterator_position(i2)); FIXME: expect warning $FL.begin() + 1

  clang_analyzer_express(clang_analyzer_container_end(FL)); // expected-warning{{$FL.end()}}
  clang_analyzer_express(clang_analyzer_iterator_position(i1)); // expected-warning{{$FL.end()}}
}

void forward_list_emplace_after_behind_begin(std::forward_list<int> &FL,
                                             int n) {
  auto i0 = FL.cbegin(), i1 = ++FL.cbegin(), i2 = FL.cend();

  clang_analyzer_denote(clang_analyzer_container_begin(FL), "$FL.begin()");
  clang_analyzer_denote(clang_analyzer_container_end(FL), "$FL.end()");

  auto i3 = FL.emplace_after(i1, n);

  clang_analyzer_eval(clang_analyzer_iterator_validity(i0)); //expected-warning{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i1)); //expected-warning{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i2)); //expected-warning{{TRUE}}

  clang_analyzer_express(clang_analyzer_container_begin(FL)); // expected-warning{{$FL.begin()}}
  clang_analyzer_express(clang_analyzer_iterator_position(i0)); // expected-warning{{$FL.begin()}}
  clang_analyzer_express(clang_analyzer_iterator_position(i1)); // expected-warning{{$FL.begin() + 1}}
  // clang_analyzer_express(clang_analyzer_iterator_position(i3)); FIXME: expect warning $FL.begin() + 2

  clang_analyzer_express(clang_analyzer_container_end(FL)); // expected-warning{{$FL.end()}}
  clang_analyzer_express(clang_analyzer_iterator_position(i2)); // expected-warning{{$FL.end()}}
}

void forward_list_emplace_after_unknown(std::forward_list<int> &FL, int n) {
  auto i0 = FL.cbegin(), i1 = return_any_iterator(FL.cbegin()), i2 = FL.cend();

  clang_analyzer_denote(clang_analyzer_container_begin(FL), "$FL.begin()");
  clang_analyzer_denote(clang_analyzer_container_end(FL), "$FL.end()");
  clang_analyzer_denote(clang_analyzer_iterator_position(i1), "$i1");

  auto i3 = FL.emplace_after(i1, n);

  clang_analyzer_eval(clang_analyzer_iterator_validity(i0)); //expected-warning{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i1)); //expected-warning{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i2)); //expected-warning{{TRUE}}

  clang_analyzer_express(clang_analyzer_container_begin(FL)); // expected-warning{{$FL.begin()}}
  clang_analyzer_express(clang_analyzer_iterator_position(i0)); // expected-warning{{$FL.begin()}}

  clang_analyzer_express(clang_analyzer_iterator_position(i1)); // expected-warning{{$i1}}
  // clang_analyzer_express(clang_analyzer_iterator_position(i3)); FIXME: expect warning $i1 + 1

  clang_analyzer_express(clang_analyzer_container_end(FL)); // expected-warning{{$FL.end()}}
  clang_analyzer_express(clang_analyzer_iterator_position(i2)); // expected-warning{{$FL.end()}}
}

/// erase()
///
/// - Design decision: shifts positions to the ->RIGHT-> (i.e. all iterator
///                    ahead of the ereased element are incremented; if the
///                    relation between the position of the erased element
///                    and the first position of the container is known, the
///                    first position of the container is also incremented).
///
/// - Iterator invalidation rules depend the container type.

/// std::list-like containers: Iterators to the erased element are invalidated.
///                            Other iterators are not affected.

void list_erase_begin(std::list<int> &L) {
  auto i0 = L.cbegin(), i1 = ++L.cbegin(), i2 = L.cend();

  clang_analyzer_denote(clang_analyzer_container_begin(L), "$L.begin()");
  clang_analyzer_denote(clang_analyzer_container_end(L), "$L.end()");

  auto i3 = L.erase(i0);

  clang_analyzer_eval(clang_analyzer_iterator_validity(i0)); //expected-warning{{FALSE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i1)); //expected-warning{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i2)); //expected-warning{{TRUE}}

  clang_analyzer_express(clang_analyzer_container_begin(L)); // expected-warning{{$L.begin()}} FIXME: Should be$L.begin() + 1
  clang_analyzer_express(clang_analyzer_iterator_position(i1)); // expected-warning{{$L.begin() + 1}}
  // clang_analyzer_express(clang_analyzer_iterator_position(i3)); FIXME: expect warning $L.begin() + 1

  clang_analyzer_express(clang_analyzer_container_end(L)); // expected-warning{{$L.end()}}
  clang_analyzer_express(clang_analyzer_iterator_position(i2)); // expected-warning{{$L.end()}}
}

void list_erase_behind_begin(std::list<int> &L, int n) {
  auto i0 = L.cbegin(), i1 = ++L.cbegin(), i2 = L.cend();

  clang_analyzer_denote(clang_analyzer_container_begin(L), "$L.begin()");
  clang_analyzer_denote(clang_analyzer_container_end(L), "$L.end()");

  auto i3 = L.erase(i1);

  clang_analyzer_eval(clang_analyzer_iterator_validity(i0)); //expected-warning{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i1)); //expected-warning{{FALSE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i2)); //expected-warning{{TRUE}}

  clang_analyzer_express(clang_analyzer_container_begin(L)); // expected-warning{{$L.begin()}} FIXME: Should be $L.begin() + 1
  clang_analyzer_express(clang_analyzer_iterator_position(i0)); // expected-warning{{$L.begin()}} FIXME: Should be $L.begin() + 1
  // clang_analyzer_express(clang_analyzer_iterator_position(i3)); FIXME: expect warning $L.begin() + 2

  clang_analyzer_express(clang_analyzer_container_end(L)); // expected-warning{{$L.end()}}
  clang_analyzer_express(clang_analyzer_iterator_position(i2)); // expected-warning{{$L.end()}}
}

void list_erase_unknown(std::list<int> &L) {
  auto i0 = L.cbegin(), i1 = return_any_iterator(L.cbegin()), i2 = L.cend();

  clang_analyzer_denote(clang_analyzer_container_begin(L), "$L.begin()");
  clang_analyzer_denote(clang_analyzer_container_end(L), "$L.end()");
  clang_analyzer_denote(clang_analyzer_iterator_position(i1), "$i1");

  auto i3 = L.erase(i1);

  clang_analyzer_eval(clang_analyzer_iterator_validity(i0)); //expected-warning{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i1)); //expected-warning{{FALSE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i2)); //expected-warning{{TRUE}}

  clang_analyzer_express(clang_analyzer_container_begin(L)); // expected-warning{{$L.begin()}}
  clang_analyzer_express(clang_analyzer_iterator_position(i0)); // expected-warning{{$L.begin()}}

  // clang_analyzer_express(clang_analyzer_iterator_position(i3)); FIXME: expect warning $i1 + 1

  clang_analyzer_express(clang_analyzer_container_end(L)); // expected-warning{{$L.end()}}
  clang_analyzer_express(clang_analyzer_iterator_position(i2)); // expected-warning{{$L.end()}}
}

void list_erase_ahead_of_end(std::list<int> &L) {
  auto i0 = L.cbegin(), i1 = --L.cend(), i2 = L.cend();

  clang_analyzer_denote(clang_analyzer_container_begin(L), "$L.begin()");
  clang_analyzer_denote(clang_analyzer_container_end(L), "$L.end()");

  auto i3 = L.erase(i1);

  clang_analyzer_eval(clang_analyzer_iterator_validity(i0)); //expected-warning{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i1)); //expected-warning{{FALSE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i2)); //expected-warning{{TRUE}}

  clang_analyzer_express(clang_analyzer_container_begin(L)); // expected-warning{{$L.begin()}}
  clang_analyzer_express(clang_analyzer_iterator_position(i0)); // expected-warning{{$L.begin()}}

  clang_analyzer_express(clang_analyzer_container_end(L)); // expected-warning{{$L.end()}}
  clang_analyzer_express(clang_analyzer_iterator_position(i2)); // expected-warning{{$L.end()}}
  // clang_analyzer_express(clang_analyzer_iterator_position(i3)); FIXME: expect warning $L.end()
}

/// std::vector-like containers: Invalidates iterators at or after the point of
///                              the erase, including the past-the-end iterator.

void vector_erase_begin(std::vector<int> &V) {
  auto i0 = V.cbegin(), i1 = ++V.cbegin(), i2 = V.cend();

  clang_analyzer_denote(clang_analyzer_container_begin(V), "$V.begin()");
  clang_analyzer_denote(clang_analyzer_container_end(V), "$V.end()");

  auto i3 = V.erase(i0);

  clang_analyzer_eval(clang_analyzer_iterator_validity(i0)); //expected-warning{{FALSE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i1)); //expected-warning{{FALSE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i2)); //expected-warning{{FALSE}}

  clang_analyzer_express(clang_analyzer_container_begin(V)); // expected-warning{{$V.begin()}} FIXME: Should be $V.begin() + 1
  // clang_analyzer_express(clang_analyzer_iterator_position(i3)); FIXME: expect warning $V.begin() + 1

  // clang_analyzer_express(clang_analyzer_container_end(V)); FIXME: expect warning $V.end()
}

void vector_erase_behind_begin(std::vector<int> &V, int n) {
  auto i0 = V.cbegin(), i1 = ++V.cbegin(), i2 = V.cend();

  clang_analyzer_denote(clang_analyzer_container_begin(V), "$V.begin()");
  clang_analyzer_denote(clang_analyzer_container_end(V), "$V.end()");

  auto i3 = V.erase(i1);

  clang_analyzer_eval(clang_analyzer_iterator_validity(i0)); //expected-warning{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i1)); //expected-warning{{FALSE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i2)); //expected-warning{{FALSE}}

  clang_analyzer_express(clang_analyzer_container_begin(V)); // expected-warning{{$V.begin()}} FIXME: Should be $V.begin() + 1
  clang_analyzer_express(clang_analyzer_iterator_position(i0)); // expected-warning{{$V.begin()}} FIXME: Should be $V.begin() + 1
  // clang_analyzer_express(clang_analyzer_iterator_position(i3)); FIXME: expect warning $V.begin() + 2

  // clang_analyzer_express(clang_analyzer_container_end(V)); FIXME: expect warning $V.end()
}

void vector_erase_unknown(std::vector<int> &V) {
  auto i0 = V.cbegin(), i1 = return_any_iterator(V.cbegin()), i2 = V.cend();

  clang_analyzer_denote(clang_analyzer_container_begin(V), "$V.begin()");
  clang_analyzer_denote(clang_analyzer_container_end(V), "$V.end()");
  clang_analyzer_denote(clang_analyzer_iterator_position(i1), "$i1");

  auto i3 = V.erase(i1);

  clang_analyzer_eval(clang_analyzer_iterator_validity(i0)); //expected-warning{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i1)); //expected-warning{{FALSE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i2)); //expected-warning{{FALSE}}

  clang_analyzer_express(clang_analyzer_container_begin(V)); // expected-warning{{$V.begin()}}
  clang_analyzer_express(clang_analyzer_iterator_position(i0)); // expected-warning{{$V.begin()}}

  // clang_analyzer_express(clang_analyzer_iterator_position(i3)); FIXME: expect warning $i1 + 1

  // clang_analyzer_express(clang_analyzer_container_end(V)); FIXME: expect warning $V.end()
}

void vector_erase_ahead_of_end(std::vector<int> &V) {
  auto i0 = V.cbegin(), i1 = --V.cend(), i2 = V.cend();

  clang_analyzer_denote(clang_analyzer_container_begin(V), "$V.begin()");
  clang_analyzer_denote(clang_analyzer_container_end(V), "$V.end()");

  auto i3 = V.erase(i1);

  clang_analyzer_eval(clang_analyzer_iterator_validity(i0)); //expected-warning{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i1)); //expected-warning{{FALSE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i2)); //expected-warning{{FALSE}}

  clang_analyzer_express(clang_analyzer_container_begin(V)); // expected-warning{{$V.begin()}}
  clang_analyzer_express(clang_analyzer_iterator_position(i0)); // expected-warning{{$V.begin()}}

  // clang_analyzer_express(clang_analyzer_container_end(V)); FIXME: expect warning $V.end()
  // clang_analyzer_express(clang_analyzer_iterator_position(i3)); FIXME: expect warning $V.end()
}

/// std::deque-like containers: All iterators are invalidated, unless the erased
///                             element is at the end or the beginning of the
///                             container, in which case only the iterators to
///                             the erased element are invalidated. The
///                             past-the-end iterator is also invalidated unless
///                             the erased element is at the beginning of the
///                             container and the last element is not erased.

void deque_erase_begin(std::deque<int> &D) {
  auto i0 = D.cbegin(), i1 = ++D.cbegin(), i2 = D.cend();

  clang_analyzer_denote(clang_analyzer_container_begin(D), "$D.begin()");
  clang_analyzer_denote(clang_analyzer_container_end(D), "$D.end()");

  auto i3 = D.erase(i0);

  clang_analyzer_eval(clang_analyzer_iterator_validity(i0)); //expected-warning{{FALSE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i1)); //expected-warning{{FALSE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i2)); //expected-warning{{FALSE}}

  clang_analyzer_express(clang_analyzer_container_begin(D)); // expected-warning{{$D.begin()}} FIXME: Should be $D.begin() + 1
  // clang_analyzer_express(clang_analyzer_iterator_position(i3)); FIXME: expect warning $D.begin() + 1

  // clang_analyzer_express(clang_analyzer_container_end(D)); FIXME: expect warning{{$D.end()
}

void deque_erase_behind_begin(std::deque<int> &D, int n) {
  auto i0 = D.cbegin(), i1 = ++D.cbegin(), i2 = D.cend();

  clang_analyzer_denote(clang_analyzer_container_begin(D), "$D.begin()");
  clang_analyzer_denote(clang_analyzer_container_end(D), "$D.end()");

  auto i3 = D.erase(i1);

  clang_analyzer_eval(clang_analyzer_iterator_validity(i0)); //expected-warning{{FALSE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i1)); //expected-warning{{FALSE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i2)); //expected-warning{{FALSE}}

  clang_analyzer_express(clang_analyzer_container_begin(D)); // expected-warning{{$D.begin()}} FIXME: Should be $D.begin() + 1
  // clang_analyzer_express(clang_analyzer_iterator_position(i3)); FIXME: expect warning $D.begin() + 2

  // clang_analyzer_express(clang_analyzer_container_end(D)); FIXME: expect warning $D.end()
}

void deque_erase_unknown(std::deque<int> &D) {
  auto i0 = D.cbegin(), i1 = return_any_iterator(D.cbegin()), i2 = D.cend();

  clang_analyzer_denote(clang_analyzer_container_begin(D), "$D.begin()");
  clang_analyzer_denote(clang_analyzer_container_end(D), "$D.end()");
  clang_analyzer_denote(clang_analyzer_iterator_position(i1), "$i1");

  auto i3 = D.erase(i1);

  clang_analyzer_eval(clang_analyzer_iterator_validity(i0)); //expected-warning{{FALSE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i1)); //expected-warning{{FALSE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i2)); //expected-warning{{FALSE}}

  clang_analyzer_express(clang_analyzer_container_begin(D)); // expected-warning{{$D.begin()}}

  // clang_analyzer_express(clang_analyzer_iterator_position(i3)); FIXME: expect warning $i1 + 1

  // clang_analyzer_express(clang_analyzer_container_end(D)); FIXME: expect warning $D.end()
}

void deque_erase_ahead_of_end(std::deque<int> &D) {
  auto i0 = D.cbegin(), i1 = --D.cend(), i2 = D.cend();

  clang_analyzer_denote(clang_analyzer_container_begin(D), "$D.begin()");
  clang_analyzer_denote(clang_analyzer_container_end(D), "$D.end()");

  auto i3 = D.erase(i1);

  clang_analyzer_eval(clang_analyzer_iterator_validity(i0)); //expected-warning{{FALSE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i1)); //expected-warning{{FALSE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i2)); //expected-warning{{FALSE}}

  clang_analyzer_express(clang_analyzer_container_begin(D)); // expected-warning{{$D.begin()}}

  // clang_analyzer_express(clang_analyzer_container_end(D)); FIXME: expect warning $D.end()
  // clang_analyzer_express(clang_analyzer_iterator_position(i3)); FIXME: expect warning $D.end()
}

/// erase_after()   [std::forward_list-like containers]
///
/// - Design decision: shifts positions to the <-LEFT<- (i.e. all iterator
///                    begind of the ereased element are decremented; if the
///                    relation between the position of the erased element
///                    and the past-the-end position of the container is known,
///                    the past-the-end position of the container is also
///                    decremented).
///
/// - Iterators to the erased element are invalidated. Other iterators are not
///   affected.


void forward_list_erase_after_begin(std::forward_list<int> &FL) {
  auto i0 = FL.cbegin(), i1 = ++FL.cbegin(), i2 = i1, i3 = FL.cend();
  ++i2;

  clang_analyzer_denote(clang_analyzer_container_begin(FL), "$FL.begin()");
  clang_analyzer_denote(clang_analyzer_container_end(FL), "$FL.end()");

  auto i4 = FL.erase_after(i0);

  clang_analyzer_eval(clang_analyzer_iterator_validity(i0)); //expected-warning{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i1)); //expected-warning{{FALSE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i2)); //expected-warning{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i3)); //expected-warning{{TRUE}}

  clang_analyzer_express(clang_analyzer_container_begin(FL)); // expected-warning{{$FL.begin()}}
  clang_analyzer_express(clang_analyzer_iterator_position(i0)); // expected-warning{{$FL.begin()}}
  clang_analyzer_express(clang_analyzer_iterator_position(i2)); // expected-warning{{$FL.begin() + 2}} FIXME: Should be $FL.begin() + 1
  // clang_analyzer_express(clang_analyzer_iterator_position(i4)); FIXME: expect warning $FL.begin() + 1

  clang_analyzer_express(clang_analyzer_container_end(FL)); // expected-warning{{$FL.end()}}
  clang_analyzer_express(clang_analyzer_iterator_position(i3)); // expected-warning{{$FL.end()}}
}

void forward_list_erase_after_unknown(std::forward_list<int> &FL) {
  auto i0 = FL.cbegin(), i1 = return_any_iterator(FL.cbegin()), i2 = i1,
    i3 = i1, i4 = FL.cend();
  ++i2;
  ++i3;
  ++i3;

  clang_analyzer_denote(clang_analyzer_container_begin(FL), "$FL.begin()");
  clang_analyzer_denote(clang_analyzer_container_end(FL), "$FL.end()");
  clang_analyzer_denote(clang_analyzer_iterator_position(i1), "$i1");

  auto i5 = FL.erase_after(i1);

  clang_analyzer_eval(clang_analyzer_iterator_validity(i0)); //expected-warning{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i1)); //expected-warning{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i2)); //expected-warning{{FALSE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i3)); //expected-warning{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_validity(i4)); //expected-warning{{TRUE}}

  clang_analyzer_express(clang_analyzer_container_begin(FL)); // expected-warning{{$FL.begin()}}
  clang_analyzer_express(clang_analyzer_iterator_position(i0)); // expected-warning{{$FL.begin()}}

  clang_analyzer_express(clang_analyzer_iterator_position(i1)); // expected-warning{{$i1}}
  clang_analyzer_express(clang_analyzer_iterator_position(i3)); // expected-warning{{$i1 + 2}} FIXME: Should be $i1 + 1
  // clang_analyzer_express(clang_analyzer_iterator_position(i5)); FIXME: expect warning $i1 + 1

  clang_analyzer_express(clang_analyzer_container_end(FL)); // expected-warning{{$FL.end()}}
  clang_analyzer_express(clang_analyzer_iterator_position(i4)); // expected-warning{{$FL.end()}}
}

struct simple_iterator_base {
  simple_iterator_base();
  simple_iterator_base(const simple_iterator_base& rhs);
  simple_iterator_base &operator=(const simple_iterator_base& rhs);
  virtual ~simple_iterator_base();
  bool friend operator==(const simple_iterator_base &lhs,
                         const simple_iterator_base &rhs);
  bool friend operator!=(const simple_iterator_base &lhs,
                         const simple_iterator_base &rhs);
private:
  int *ptr;
};

struct simple_derived_iterator: public simple_iterator_base {
  int& operator*();
  int* operator->();
  simple_iterator_base &operator++();
  simple_iterator_base operator++(int);
  simple_iterator_base &operator--();
  simple_iterator_base operator--(int);
};

struct simple_container {
  typedef simple_derived_iterator iterator;

  iterator begin();
  iterator end();
};

void good_derived(simple_container c) {
  auto i0 = c.end();
  
  if (i0 != c.end()) {
    clang_analyzer_warnIfReached();
  }
}

void iter_diff(std::vector<int> &V) {
  auto i0 = V.begin(), i1 = V.end();
  ptrdiff_t len = i1 - i0; // no-crash
}

void deferred_assumption(std::vector<int> &V, int e) {
  const auto first = V.begin();
  const auto comp1 = (first != V.end()), comp2 = (first == V.end());
  if (comp1) {
    clang_analyzer_eval(clang_analyzer_container_end(V) ==
                        clang_analyzer_iterator_position(first)); // expected-warning@-1{{FALSE}}
  }
}

void loop(std::vector<int> &V, int e) {
  auto start = V.begin();
  while (true) {
    auto item = std::find(start, V.end(), e);
    if (item == V.end())
      break;

    clang_analyzer_eval(clang_analyzer_container_end(V) ==
                        clang_analyzer_iterator_position(item)); // expected-warning@-1{{FALSE}}
  }
}

template <typename InputIterator, typename T>
InputIterator nonStdFind(InputIterator first, InputIterator last,
                         const T &val) {
  for (auto i = first; i != last; ++i) {
    if (*i == val) {
      return i;
    }
  }
  return last;
}

void non_std_find(std::vector<int> &V, int e) {
  auto first = nonStdFind(V.begin(), V.end(), e);
  clang_analyzer_eval(clang_analyzer_container_end(V) ==
                      clang_analyzer_iterator_position(first)); // expected-warning@-1{{FALSE}} expected-warning@-1{{TRUE}}
  if (V.end() != first) {
    clang_analyzer_eval(clang_analyzer_container_end(V) ==
                        clang_analyzer_iterator_position(first)); // expected-warning@-1{{FALSE}}
  }
}

void clang_analyzer_printState();

void print_state(std::vector<int> &V) {
  const auto i0 = V.cbegin();
  clang_analyzer_printState();

// CHECK:      "checker_messages": [
// CHECK-NEXT:   { "checker": "alpha.cplusplus.IteratorModeling", "messages": [
// CHECK-NEXT:     "Container Data :",
// CHECK-NEXT:     "SymRegion{reg_$[[#]]<std::vector<int> & V>} : [ conj_$[[#]]{long, LC[[#]], S[[#]], #[[#]]} .. <Unknown> ]",
// CHECK-NEXT:     "Iterator Positions :",
// CHECK-NEXT:     "i0 : Valid ; Container == SymRegion{reg_$[[#]]<std::vector<int> & V>} ; Offset == conj_$[[#]]{long, LC[[#]], S[[#]], #[[#]]}"
// CHECK-NEXT:   ]}

  const auto i1 = V.cend();
  clang_analyzer_printState();
  
// CHECK:      "checker_messages": [
// CHECK-NEXT:   { "checker": "alpha.cplusplus.IteratorModeling", "messages": [
// CHECK-NEXT:     "Container Data :",
// CHECK-NEXT:     "SymRegion{reg_$[[#]]<std::vector<int> & V>} : [ conj_$[[#]]{long, LC[[#]], S[[#]], #[[#]]} .. conj_$[[#]]{long, LC[[#]], S[[#]], #[[#]]} ]",
// CHECK-NEXT:     "Iterator Positions :",
// CHECK-NEXT:     "i1 : Valid ; Container == SymRegion{reg_$[[#]]<std::vector<int> & V>} ; Offset == conj_$[[#]]{long, LC[[#]], S[[#]], #[[#]]}"
// CHECK-NEXT:   ]}
}
