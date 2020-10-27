// RUN: %clang_analyze_cc1 -std=c++11 -analyzer-checker=core,cplusplus,alpha.cplusplus.IteratorRange -analyzer-config aggressive-binary-operation-simplification=true -analyzer-config c++-container-inlining=false -analyzer-output=text %s -verify
// RUN: %clang_analyze_cc1 -std=c++11 -analyzer-checker=core,cplusplus,alpha.cplusplus.IteratorRange -analyzer-config aggressive-binary-operation-simplification=true -analyzer-config c++-container-inlining=true -DINLINE=1 -analyzer-output=text %s -verify

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
      // expected-note@-1{{Past-the-end iterator dereferenced}}
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
       // expected-note@-1{{Iterator incremented behind the past-the-end iterator}}
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
       // expected-note@-1{{Iterator incremented behind the past-the-end iterator}}
}

// Prefix decrement - operator--()

void decr_begin(const std::vector<int> &V) {
  auto i = V.begin();
  --i; // expected-warning{{Iterator decremented ahead of its valid range}}
       // expected-note@-1{{Iterator decremented ahead of its valid range}}
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
       // expected-note@-1{{Iterator decremented ahead of its valid range}}
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
          // expected-note@-1{{Iterator incremented behind the past-the-end iterator}}
}

void incr_by_2_end(const std::vector<int> &V) {
  auto i = V.end();
  i += 2; // expected-warning{{Iterator incremented behind the past-the-end iterator}}
          // expected-note@-1{{Iterator incremented behind the past-the-end iterator}}
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
                  // expected-note@-1{{Iterator incremented behind the past-the-end iterator}}
}

void incr_by_2_copy_end(const std::vector<int> &V) {
  auto i = V.end();
  auto j = i + 2; // expected-warning{{Iterator incremented behind the past-the-end iterator}}
                  // expected-note@-1{{Iterator incremented behind the past-the-end iterator}}
}

// Subtraction assignment - operator-=(int)

void decr_by_2_begin(const std::vector<int> &V) {
  auto i = V.begin();
  i -= 2; // expected-warning{{Iterator decremented ahead of its valid range}}
          // expected-note@-1{{Iterator decremented ahead of its valid range}}
}

void decr_by_2_behind_begin(const std::vector<int> &V) {
  auto i = ++V.begin();
  i -= 2; // expected-warning{{Iterator decremented ahead of its valid range}}
          // expected-note@-1{{Iterator decremented ahead of its valid range}}
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
                  // expected-note@-1{{Iterator decremented ahead of its valid range}}
}

void decr_by_2_copy_behind_begin(const std::vector<int> &V) {
  auto i = ++V.begin();
  auto j = i - 2; // expected-warning{{Iterator decremented ahead of its valid range}}
                  // expected-note@-1{{Iterator decremented ahead of its valid range}}
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
                 // expected-note@-1{{Past-the-end iterator dereferenced}}
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
  auto j = i[-1]; // expected-warning{{Past-the-end iterator dereferenced}} FIXME: expect no warning
                  // expected-note@-1{{Past-the-end iterator dereferenced}}
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
                 // expected-note@-1{{Past-the-end iterator dereferenced}} FIXME: expect note@-1 Iterator incremented behind the past-the-end iterator
}

//
// std::advance()
//

// std::advance() by +1

void advance_plus_1_begin(const std::vector<int> &V) {
  auto i = V.begin();
  std::advance(i, 1); // no-warning
}

void advance_plus_1_behind_begin(const std::vector<int> &V) {
  auto i = ++V.begin();
  std::advance(i, 1); // no-warning
}

void advance_plus_1_unknown(const std::vector<int> &V) {
  auto i = return_any_iterator(V.begin());
  std::advance(i, 1); // no-warning
}

void advance_plus_1_ahead_of_end(const std::vector<int> &V) {
  auto i = --V.end();
  std::advance(i, 1); // no-warning
}

void advance_plus_1_end(const std::vector<int> &V) {
  auto i = V.end();
  std::advance(i, 1); // expected-warning{{Iterator incremented behind the past-the-end iterator}}
                      // expected-note@-1{{Iterator incremented behind the past-the-end iterator}}
}

// std::advance() by -1

void advance_minus_1_begin(const std::vector<int> &V) {
  auto i = V.begin();
  std::advance(i, -1); // expected-warning{{Iterator decremented ahead of its valid range}}
                       // expected-note@-1{{Iterator decremented ahead of its valid range}}
}

void advance_minus_1_behind_begin(const std::vector<int> &V) {
  auto i = ++V.begin();
  std::advance(i, -1); // no-warning
}

void advance_minus_1_unknown(const std::vector<int> &V) {
  auto i = return_any_iterator(V.begin());
  std::advance(i, -1); // no-warning
}

void advance_minus_1_ahead_of_end(const std::vector<int> &V) {
  auto i = --V.end();
  std::advance(i, -1); // no-warning
}

void advance_minus_1_end(const std::vector<int> &V) {
  auto i = V.end();
  std::advance(i, -1); // no-warning
}

// std::advance() by +2

void advance_plus_2_begin(const std::vector<int> &V) {
  auto i = V.begin();
  std::advance(i, 2); // no-warning
}

void advance_plus_2_behind_begin(const std::vector<int> &V) {
  auto i = ++V.begin();
  std::advance(i, 2); // no-warning
}

void advance_plus_2_unknown(const std::vector<int> &V) {
  auto i = return_any_iterator(V.begin());
  std::advance(i, 2); // no-warning
}

void advance_plus_2_ahead_of_end(const std::vector<int> &V) {
  auto i = --V.end();
  std::advance(i, 2); // expected-warning{{Iterator incremented behind the past-the-end iterator}}
                      // expected-note@-1{{Iterator incremented behind the past-the-end iterator}}
}

void advance_plus_2_end(const std::vector<int> &V) {
  auto i = V.end();
  std::advance(i, 2); // expected-warning{{Iterator incremented behind the past-the-end iterator}}
                      // expected-note@-1{{Iterator incremented behind the past-the-end iterator}}
}

// std::advance() by -2

void advance_minus_2_begin(const std::vector<int> &V) {
  auto i = V.begin();
  std::advance(i, -2); // expected-warning{{Iterator decremented ahead of its valid range}}
                       // expected-note@-1{{Iterator decremented ahead of its valid range}}
}

void advance_minus_2_behind_begin(const std::vector<int> &V) {
  auto i = ++V.begin();
  std::advance(i, -2); // expected-warning{{Iterator decremented ahead of its valid range}}
                       // expected-note@-1{{Iterator decremented ahead of its valid range}}
}

void advance_minus_2_unknown(const std::vector<int> &V) {
  auto i = return_any_iterator(V.begin());
  std::advance(i, -2); // no-warning
}

void advance_minus_2_ahead_of_end(const std::vector<int> &V) {
  auto i = --V.end();
  std::advance(i, -2); // no-warning
}

void advance_minus_2_end(const std::vector<int> &V) {
  auto i = V.end();
  std::advance(i, -2); // no-warning
}

// std::advance() by 0

void advance_0_begin(const std::vector<int> &V) {
  auto i = V.begin();
  std::advance(i, 0); // no-warning
}

void advance_0_behind_begin(const std::vector<int> &V) {
  auto i = ++V.begin();
  std::advance(i, 0); // no-warning
}

void advance_0_unknown(const std::vector<int> &V) {
  auto i = return_any_iterator(V.begin());
  std::advance(i, 0); // no-warning
}

void advance_0_ahead_of_end(const std::vector<int> &V) {
  auto i = --V.end();
  std::advance(i, 0); // no-warning
}

void advance_0_end(const std::vector<int> &V) {
  auto i = V.end();
  std::advance(i, 0); // no-warning
}

//
// std::next()
//

// std::next() by +1 (default)

void next_plus_1_begin(const std::vector<int> &V) {
  auto i = V.begin();
  auto j = std::next(i); // no-warning
}

void next_plus_1_behind_begin(const std::vector<int> &V) {
  auto i = ++V.begin();
  auto j = std::next(i); // no-warning
}

void next_plus_1_unknown(const std::vector<int> &V) {
  auto i = return_any_iterator(V.begin());
  auto j = std::next(i); // no-warning
}

void next_plus_1_ahead_of_end(const std::vector<int> &V) {
  auto i = --V.end();
  auto j = std::next(i); // no-warning
}

void next_plus_1_end(const std::vector<int> &V) {
  auto i = V.end();
  auto j = std::next(i); // expected-warning{{Iterator incremented behind the past-the-end iterator}}
                         // expected-note@-1{{Iterator incremented behind the past-the-end iterator}}
}

// std::next() by -1

void next_minus_1_begin(const std::vector<int> &V) {
  auto i = V.begin();
  auto j = std::next(i, -1); // expected-warning{{Iterator decremented ahead of its valid range}}
                             // expected-note@-1{{Iterator decremented ahead of its valid range}}
}

void next_minus_1_behind_begin(const std::vector<int> &V) {
  auto i = ++V.begin();
  auto j = std::next(i, -1); // no-warning
}

void next_minus_1_unknown(const std::vector<int> &V) {
  auto i = return_any_iterator(V.begin());
  auto j = std::next(i, -1); // no-warning
}

void next_minus_1_ahead_of_end(const std::vector<int> &V) {
  auto i = --V.end();
  auto j = std::next(i, -1); // no-warning
}

void next_minus_1_end(const std::vector<int> &V) {
  auto i = V.end();
  auto j = std::next(i, -1); // no-warning
}

// std::next() by +2

void next_plus_2_begin(const std::vector<int> &V) {
  auto i = V.begin();
  auto j = std::next(i, 2); // no-warning
}

void next_plus_2_behind_begin(const std::vector<int> &V) {
  auto i = ++V.begin();
  auto j = std::next(i, 2); // no-warning
}

void next_plus_2_unknown(const std::vector<int> &V) {
  auto i = return_any_iterator(V.begin());
  auto j = std::next(i, 2); // no-warning
}

void next_plus_2_ahead_of_end(const std::vector<int> &V) {
  auto i = --V.end();
  auto j = std::next(i, 2); // expected-warning{{Iterator incremented behind the past-the-end iterator}}
                            // expected-note@-1{{Iterator incremented behind the past-the-end iterator}}
}

void next_plus_2_end(const std::vector<int> &V) {
  auto i = V.end();
  auto j = std::next(i, 2); // expected-warning{{Iterator incremented behind the past-the-end iterator}}
                            // expected-note@-1{{Iterator incremented behind the past-the-end iterator}}
}

// std::next() by -2

void next_minus_2_begin(const std::vector<int> &V) {
  auto i = V.begin();
  auto j = std::next(i, -2); // expected-warning{{Iterator decremented ahead of its valid range}}
                             // expected-note@-1{{Iterator decremented ahead of its valid range}}
}

void next_minus_2_behind_begin(const std::vector<int> &V) {
  auto i = ++V.begin();
  auto j = std::next(i, -2); // expected-warning{{Iterator decremented ahead of its valid range}}
                             // expected-note@-1{{Iterator decremented ahead of its valid range}}
}

void next_minus_2_unknown(const std::vector<int> &V) {
  auto i = return_any_iterator(V.begin());
  auto j = std::next(i, -2); // no-warning
}

void next_minus_2_ahead_of_end(const std::vector<int> &V) {
  auto i = --V.end();
  auto j = std::next(i, -2); // no-warning
}

void next_minus_2_end(const std::vector<int> &V) {
  auto i = V.end();
  auto j = std::next(i, -2); // no-warning
}

// std::next() by 0

void next_0_begin(const std::vector<int> &V) {
  auto i = V.begin();
  auto j = std::next(i, 0); // no-warning
}

void next_0_behind_begin(const std::vector<int> &V) {
  auto i = ++V.begin();
  auto j = std::next(i, 0); // no-warning
}

void next_0_unknown(const std::vector<int> &V) {
  auto i = return_any_iterator(V.begin());
  auto j = std::next(i, 0); // no-warning
}

void next_0_ahead_of_end(const std::vector<int> &V) {
  auto i = --V.end();
  auto j = std::next(i, 0); // no-warning
}

void next_0_end(const std::vector<int> &V) {
  auto i = V.end();
  auto j = std::next(i, 0); // no-warning
}

//
// std::prev()
//

// std::prev() by +1 (default)

void prev_plus_1_begin(const std::vector<int> &V) {
  auto i = V.begin();
  auto j = std::prev(i); // expected-warning{{Iterator decremented ahead of its valid range}}
                         // expected-note@-1{{Iterator decremented ahead of its valid range}}
}

void prev_plus_1_behind_begin(const std::vector<int> &V) {
  auto i = ++V.begin();
  auto j = std::prev(i); // no-warning
}

void prev_plus_1_unknown(const std::vector<int> &V) {
  auto i = return_any_iterator(V.begin());
  auto j = std::prev(i); // no-warning
}

void prev_plus_1_ahead_of_end(const std::vector<int> &V) {
  auto i = --V.end();
  auto j = std::prev(i); // no-warning
}

void prev_plus_1_end(const std::vector<int> &V) {
  auto i = V.end();
  auto j = std::prev(i); // no-warning
}

// std::prev() by -1

void prev_minus_1_begin(const std::vector<int> &V) {
  auto i = V.begin();
  auto j = std::prev(i, -1); // no-warning
}

void prev_minus_1_behind_begin(const std::vector<int> &V) {
  auto i = ++V.begin();
  auto j = std::prev(i, -1); // no-warning
}

void prev_minus_1_unknown(const std::vector<int> &V) {
  auto i = return_any_iterator(V.begin());
  auto j = std::prev(i, -1); // no-warning
}

void prev_minus_1_ahead_of_end(const std::vector<int> &V) {
  auto i = --V.end();
  auto j = std::prev(i, -1); // no-warning
}

void prev_minus_1_end(const std::vector<int> &V) {
  auto i = V.end();
  auto j = std::prev(i, -1); // expected-warning{{Iterator incremented behind the past-the-end iterator}}
                             // expected-note@-1{{Iterator incremented behind the past-the-end iterator}}
}

// std::prev() by +2

void prev_plus_2_begin(const std::vector<int> &V) {
  auto i = V.begin();
  auto j = std::prev(i, 2); // expected-warning{{Iterator decremented ahead of its valid range}}
                            // expected-note@-1{{Iterator decremented ahead of its valid range}}
}

void prev_plus_2_behind_begin(const std::vector<int> &V) {
  auto i = ++V.begin();
  auto j = std::prev(i, 2); // expected-warning{{Iterator decremented ahead of its valid range}}
                            // expected-note@-1{{Iterator decremented ahead of its valid range}}
}

void prev_plus_2_unknown(const std::vector<int> &V) {
  auto i = return_any_iterator(V.begin());
  auto j = std::prev(i, 2); // no-warning
}

void prev_plus_2_ahead_of_end(const std::vector<int> &V) {
  auto i = --V.end();
  auto j = std::prev(i, 2); // no-warning
}

void prev_plus_2_end(const std::vector<int> &V) {
  auto i = V.end();
  auto j = std::prev(i, 2); // no-warning
}

// std::prev() by -2

void prev_minus_2_begin(const std::vector<int> &V) {
  auto i = V.begin();
  auto j = std::prev(i, -2); // no-warning
}

void prev_minus_2_behind_begin(const std::vector<int> &V) {
  auto i = ++V.begin();
  auto j = std::prev(i, -2); // no-warning
}

void prev_minus_2_unknown(const std::vector<int> &V) {
  auto i = return_any_iterator(V.begin());
  auto j = std::prev(i, -2); // no-warning
}

void prev_minus_2_ahead_of_end(const std::vector<int> &V) {
  auto i = --V.end();
  auto j = std::prev(i, -2); // expected-warning{{Iterator incremented behind the past-the-end iterator}}
                             // expected-note@-1{{Iterator incremented behind the past-the-end iterator}}
}

void prev_minus_2_end(const std::vector<int> &V) {
  auto i = V.end();
  auto j = std::prev(i, -2); // expected-warning{{Iterator incremented behind the past-the-end iterator}}
                             // expected-note@-1{{Iterator incremented behind the past-the-end iterator}}
}

// std::prev() by 0

void prev_0_begin(const std::vector<int> &V) {
  auto i = V.begin();
  auto j = std::prev(i, 0); // no-warning
}

void prev_0_behind_begin(const std::vector<int> &V) {
  auto i = ++V.begin();
  auto j = std::prev(i, 0); // no-warning
}

void prev_0_unknown(const std::vector<int> &V) {
  auto i = return_any_iterator(V.begin());
  auto j = std::prev(i, 0); // no-warning
}

void prev_0_ahead_of_end(const std::vector<int> &V) {
  auto i = --V.end();
  auto j = std::prev(i, 0); // no-warning
}

void prev_0_end(const std::vector<int> &V) {
  auto i = V.end();
  auto j = std::prev(i, 0); // no-warning
}

// std::prev() with int* for checking Loc value argument
namespace std {
template <typename T>
T prev(T, int *);
}

void prev_loc_value(const std::vector<int> &V, int o) {

  auto i = return_any_iterator(V.begin());
  int *offset = &o;
  auto j = std::prev(i, offset); // no-warning
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
  int n = i->n; // expected-warning{{Past-the-end iterator dereferenced}}
                // expected-note@-1{{Past-the-end iterator dereferenced}}
}

// Container modification - test path notes

void deref_end_after_pop_back(std::vector<int> &V) {
  const auto i = --V.end();

  V.pop_back(); // expected-note{{Container 'V' shrank from the back by 1 position}}

  *i; // expected-warning{{Past-the-end iterator dereferenced}}
      // expected-note@-1{{Past-the-end iterator dereferenced}}
}

template<typename T>
struct cont_with_ptr_iterator {
  T* begin() const;
  T* end() const;
};

void deref_end_ptr_iterator(const cont_with_ptr_iterator<S> &c) {
  auto i = c.end();
  (void) *i; // expected-warning{{Past-the-end iterator dereferenced}}
             // expected-note@-1{{Past-the-end iterator dereferenced}}
}

void array_deref_end_ptr_iterator(const cont_with_ptr_iterator<S> &c) {
  auto i = c.end();
  (void) i[0]; // expected-warning{{Past-the-end iterator dereferenced}}
               // expected-note@-1{{Past-the-end iterator dereferenced}}
}

void arrow_deref_end_ptr_iterator(const cont_with_ptr_iterator<S> &c) {
  auto i = c.end();
  (void) i->n; // expected-warning{{Past-the-end iterator dereferenced}}
               // expected-note@-1{{Past-the-end iterator dereferenced}}
}

void arrow_star_deref_end_ptr_iterator(const cont_with_ptr_iterator<S> &c,
                                       int S::*p) {
  auto i = c.end();
  (void)(i->*p); // expected-warning{{Past-the-end iterator dereferenced}}
                 // expected-note@-1{{Past-the-end iterator dereferenced}}
}

void prefix_incr_end_ptr_iterator(const cont_with_ptr_iterator<S> &c) {
  auto i = c.end();
  ++i; // expected-warning{{Iterator incremented behind the past-the-end iterator}}
       // expected-note@-1{{Iterator incremented behind the past-the-end iterator}}
}

void postfix_incr_end_ptr_iterator(const cont_with_ptr_iterator<S> &c) {
  auto i = c.end();
  i++; // expected-warning{{Iterator incremented behind the past-the-end iterator}}
       // expected-note@-1{{Iterator incremented behind the past-the-end iterator}}
}

void prefix_decr_begin_ptr_iterator(const cont_with_ptr_iterator<S> &c) {
  auto i = c.begin();
  --i; // expected-warning{{Iterator decremented ahead of its valid range}}
       // expected-note@-1{{Iterator decremented ahead of its valid range}}
}

void postfix_decr_begin_ptr_iterator(const cont_with_ptr_iterator<S> &c) {
  auto i = c.begin();
  i--; // expected-warning{{Iterator decremented ahead of its valid range}}
       // expected-note@-1{{Iterator decremented ahead of its valid range}}
}

void prefix_add_2_end_ptr_iterator(const cont_with_ptr_iterator<S> &c) {
  auto i = c.end();
  (void)(i + 2); // expected-warning{{Iterator incremented behind the past-the-end iterator}}
                 // expected-note@-1{{Iterator incremented behind the past-the-end iterator}}
}

void postfix_add_assign_2_end_ptr_iterator(const cont_with_ptr_iterator<S> &c) {
  auto i = c.end();
  i += 2; // expected-warning{{Iterator incremented behind the past-the-end iterator}}
          // expected-note@-1{{Iterator incremented behind the past-the-end iterator}}
}

void prefix_minus_2_begin_ptr_iterator(const cont_with_ptr_iterator<S> &c) {
  auto i = c.begin();
  (void)(i - 2); // expected-warning{{Iterator decremented ahead of its valid range}}
                 // expected-note@-1{{Iterator decremented ahead of its valid range}}
}

void postfix_minus_assign_2_begin_ptr_iterator(
    const cont_with_ptr_iterator<S> &c) {
  auto i = c.begin();
  i -= 2; // expected-warning{{Iterator decremented ahead of its valid range}}
          // expected-note@-1{{Iterator decremented ahead of its valid range}}
}

void ptr_iter_diff(cont_with_ptr_iterator<S> &c) {
  auto i0 = c.begin(), i1 = c.end();
  ptrdiff_t len = i1 - i0; // no-crash
}
