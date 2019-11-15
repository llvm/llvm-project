// RUN: %clang_analyze_cc1 -std=c++11 -analyzer-checker=core,cplusplus,alpha.cplusplus.IteratorRange -analyzer-config aggressive-binary-operation-simplification=true -analyzer-config c++-container-inlining=false %s -verify
// RUN: %clang_analyze_cc1 -std=c++11 -analyzer-checker=core,cplusplus,alpha.cplusplus.IteratorRange -analyzer-config aggressive-binary-operation-simplification=true -analyzer-config c++-container-inlining=true -DINLINE=1 %s -verify

#include "Inputs/system-header-simulator-cxx.h"

void clang_analyzer_warnIfReached();

// Dereference - operator*()

void deref_begin(const std::vector<int> &V) {
  auto i = V.begin();
  *i; // no-warning
}

void deref_begind_begin(const std::vector<int> &V) {
  auto i = ++V.begin();
  *i; // no-warning
}

template <typename Iter> Iter return_any_iterator(const Iter &It);

void deref_unknown(const std::vector<int> &V) {
  auto i = return_any_iterator(V.begin());
  *i; // no-warning
}

void deref_ahead_of_end(const std::vector<int> &V) {
  auto i = --V.end();
  *i; // no-warning
}

void deref_end(const std::vector<int> &V) {
  auto i = V.end();
  *i; // expected-warning{{Past-the-end iterator dereferenced}}
}

// Prefix increment - operator++()

void incr_begin(const std::vector<int> &V) {
  auto i = V.begin();
  ++i; // no-warning
}

void incr_behind_begin(const std::vector<int> &V) {
  auto i = ++V.begin();
  ++i; // no-warning
}

void incr_unknown(const std::vector<int> &V) {
  auto i = return_any_iterator(V.begin());
  ++i; // no-warning
}

void incr_ahead_of_end(const std::vector<int> &V) {
  auto i = --V.end();
  ++i; // no-warning
}

void incr_end(const std::vector<int> &V) {
  auto i = V.end();
  ++i; // expected-warning{{Iterator incremented behind the past-the-end iterator}}
}

// Postfix increment - operator++(int)

void begin_incr(const std::vector<int> &V) {
  auto i = V.begin();
  i++; // no-warning
}

void behind_begin_incr(const std::vector<int> &V) {
  auto i = ++V.begin();
  i++; // no-warning
}

void unknown_incr(const std::vector<int> &V) {
  auto i = return_any_iterator(V.begin());
  i++; // no-warning
}

void ahead_of_end_incr(const std::vector<int> &V) {
  auto i = --V.end();
  i++; // no-warning
}

void end_incr(const std::vector<int> &V) {
  auto i = V.end();
  i++; // expected-warning{{Iterator incremented behind the past-the-end iterator}}
}

// Prefix decrement - operator--()

void decr_begin(const std::vector<int> &V) {
  auto i = V.begin();
  --i; // expected-warning{{Iterator decremented ahead of its valid range}}
}

void decr_behind_begin(const std::vector<int> &V) {
  auto i = ++V.begin();
  --i; // no-warning
}

void decr_unknown(const std::vector<int> &V) {
  auto i = return_any_iterator(V.begin());
  --i; // no-warning
}

void decr_ahead_of_end(const std::vector<int> &V) {
  auto i = --V.end();
  --i; // no-warning
}

void decr_end(const std::vector<int> &V) {
  auto i = V.end();
  --i; // no-warning
}

// Postfix decrement - operator--(int)

void begin_decr(const std::vector<int> &V) {
  auto i = V.begin();
  i--; // expected-warning{{Iterator decremented ahead of its valid range}}
}

void behind_begin_decr(const std::vector<int> &V) {
  auto i = ++V.begin();
  i--; // no-warning
}

void unknown_decr(const std::vector<int> &V) {
  auto i = return_any_iterator(V.begin());
  i--; // no-warning
}

void ahead_of_end_decr(const std::vector<int> &V) {
  auto i = --V.end();
  i--; // no-warning
}

void end_decr(const std::vector<int> &V) {
  auto i = V.end();
  i--; // no-warning
}

// Addition assignment - operator+=(int)

void incr_by_2_begin(const std::vector<int> &V) {
  auto i = V.begin();
  i += 2; // no-warning
}

void incr_by_2_behind_begin(const std::vector<int> &V) {
  auto i = ++V.begin();
  i += 2; // no-warning
}

void incr_by_2_unknown(const std::vector<int> &V) {
  auto i = return_any_iterator(V.begin());
  i += 2; // no-warning
}

void incr_by_2_ahead_by_2_of_end(const std::vector<int> &V) {
  auto i = --V.end();
  --i;
  i += 2; // no-warning
}

void incr_by_2_ahead_of_end(const std::vector<int> &V) {
  auto i = --V.end();
  i += 2; // expected-warning{{Iterator incremented behind the past-the-end iterator}}
}

void incr_by_2_end(const std::vector<int> &V) {
  auto i = V.end();
  i += 2; // expected-warning{{Iterator incremented behind the past-the-end iterator}}
}

// Addition - operator+(int)

void incr_by_2_copy_begin(const std::vector<int> &V) {
  auto i = V.begin();
  auto j = i + 2; // no-warning
}

void incr_by_2_copy_behind_begin(const std::vector<int> &V) {
  auto i = ++V.begin();
  auto j = i + 2; // no-warning
}

void incr_by_2_copy_unknown(const std::vector<int> &V) {
  auto i = return_any_iterator(V.begin());
  auto j = i + 2; // no-warning
}

void incr_by_2_copy_ahead_by_2_of_end(const std::vector<int> &V) {
  auto i = --V.end();
  --i;
  auto j = i + 2; // no-warning
}

void incr_by_2_copy_ahead_of_end(const std::vector<int> &V) {
  auto i = --V.end();
  auto j = i + 2; // expected-warning{{Iterator incremented behind the past-the-end iterator}}
}

void incr_by_2_copy_end(const std::vector<int> &V) {
  auto i = V.end();
  auto j = i + 2; // expected-warning{{Iterator incremented behind the past-the-end iterator}}
}

// Subtraction assignment - operator-=(int)

void decr_by_2_begin(const std::vector<int> &V) {
  auto i = V.begin();
  i -= 2; // expected-warning{{Iterator decremented ahead of its valid range}}
}

void decr_by_2_behind_begin(const std::vector<int> &V) {
  auto i = ++V.begin();
  i -= 2; // expected-warning{{Iterator decremented ahead of its valid range}}
}

void decr_by_2_behind_begin_by_2(const std::vector<int> &V) {
  auto i = ++V.begin();
  ++i;
  i -= 2; // no-warning
}

void decr_by_2_unknown(const std::vector<int> &V) {
  auto i = return_any_iterator(V.begin());
  i -= 2; // no-warning
}

void decr_by_2_ahead_of_end(const std::vector<int> &V) {
  auto i = --V.end();
  i -= 2; // no-warning
}

void decr_by_2_end(const std::vector<int> &V) {
  auto i = V.end();
  i -= 2; // no-warning
}

// Subtraction - operator-(int)

void decr_by_2_copy_begin(const std::vector<int> &V) {
  auto i = V.begin();
  auto j = i - 2; // expected-warning{{Iterator decremented ahead of its valid range}}
}

void decr_by_2_copy_behind_begin(const std::vector<int> &V) {
  auto i = ++V.begin();
  auto j = i - 2; // expected-warning{{Iterator decremented ahead of its valid range}}
}

void decr_by_2_copy_behind_begin_by_2(const std::vector<int> &V) {
  auto i = ++V.begin();
  ++i;
  auto j = i - 2; // no-warning
}

void decr_by_2_copy_unknown(const std::vector<int> &V) {
  auto i = return_any_iterator(V.begin());
  auto j = i - 2; // no-warning
}

void decr_by_2_copy_ahead_of_end(const std::vector<int> &V) {
  auto i = --V.end();
  auto j = i - 2; // no-warning
}

void decr_by_2_copy_end(const std::vector<int> &V) {
  auto i = V.end();
  auto j = i - 2; // no-warning
}

//
// Subscript - operator[](int)
//

// By zero

void subscript_zero_begin(const std::vector<int> &V) {
  auto i = V.begin();
  auto j = i[0]; // no-warning
}

void subscript_zero_behind_begin(const std::vector<int> &V) {
  auto i = ++V.begin();
  auto j = i[0]; // no-warning
}

void subscript_zero_unknown(const std::vector<int> &V) {
  auto i = return_any_iterator(V.begin());
  auto j = i[0]; // no-warning
}

void subscript_zero_ahead_of_end(const std::vector<int> &V) {
  auto i = --V.end();
  auto j = i[0]; // no-warning
}

void subscript_zero_end(const std::vector<int> &V) {
  auto i = V.end();
  auto j = i[0]; // expected-warning{{Past-the-end iterator dereferenced}}
}

// By negative number

void subscript_negative_begin(const std::vector<int> &V) {
  auto i = V.begin();
  auto j = i[-1]; // no-warning FIXME: expect warning Iterator decremented ahead of its valid range
}

void subscript_negative_behind_begin(const std::vector<int> &V) {
  auto i = ++V.begin();
  auto j = i[-1]; // no-warning
}

void subscript_negative_unknown(const std::vector<int> &V) {
  auto i = return_any_iterator(V.begin());
  auto j = i[-1]; // no-warning
}

void subscript_negative_ahead_of_end(const std::vector<int> &V) {
  auto i = --V.end();
  auto j = i[-1]; // no-warning
}

void subscript_negative_end(const std::vector<int> &V) {
  auto i = V.end();
  auto j = i[-1]; // // expected-warning{{Past-the-end iterator dereferenced}} FIXME: expect no warning
}

// By positive number

void subscript_positive_begin(const std::vector<int> &V) {
  auto i = V.begin();
  auto j = i[1]; // no-warning
}

void subscript_positive_behind_begin(const std::vector<int> &V) {
  auto i = ++V.begin();
  auto j = i[1]; // no-warning
}

void subscript_positive_unknown(const std::vector<int> &V) {
  auto i = return_any_iterator(V.begin());
  auto j = i[1]; // no-warning
}

void subscript_positive_ahead_of_end(const std::vector<int> &V) {
  auto i = --V.end();
  auto j = i[1]; // no-warning FIXME: expected warning Past-the-end iterator dereferenced
}

void subscript_positive_end(const std::vector<int> &V) {
  auto i = V.end();
  auto j = i[1]; // expected-warning{{Past-the-end iterator dereferenced}} FIXME: expect warning Iterator incremented behind the past-the-end iterator
}

//
// Structure member dereference operators
//

struct S {
  int n;
};

// Member dereference - operator->()

void arrow_deref_begin(const std::vector<S> &V) {
  auto i = V.begin();
  int n = i->n; // no-warning
}

void arrow_deref_end(const std::vector<S> &V) {
  auto i = V.end();
  int n = i->n; //  expected-warning{{Past-the-end iterator dereferenced}}
}
