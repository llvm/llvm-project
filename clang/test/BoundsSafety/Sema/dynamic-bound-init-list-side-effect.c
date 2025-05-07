
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s

#include <ptrcheck.h>

int getlen();
int side_effect();
void *__bidi_indexable getptr();

struct CountedByData {
  int *__counted_by(len + 1) ptr;
  int len;
};

struct SizedByData {
  void *__sized_by(len) ptr;
  unsigned len;
};

struct CountedByOrNullData {
  int *__counted_by_or_null(len + 1) ptr;
  int len;
};

struct SizedByOrNullData {
  void *__sized_by_or_null(len) ptr;
  unsigned len;
};

struct RangedData {
  int *__ended_by(iter) start;
  int *__ended_by(end) iter;
  void *end;
};

struct CountedBySizedByDataMix {
  int *__counted_by(len + 1) ptr;
  int *__sized_by(len) ptr2;
  int len;
};

struct InnerStruct {
  int value;
};
struct CountedByDataWithSubStruct {
  int *__counted_by(len + 1) ptr;
  int len;
  struct InnerStruct inner;
};

struct CountedByDataWithSubStructAtStart {
  struct InnerStruct inner;
  int *__counted_by(len + 1) ptr;
  int len;
};

struct CountedByDataWithOtherField {
  int *__counted_by(len + 1) ptr;
  int len;
  int other_data;
};



int g[10];

void TestCountedBy(void) {
  // expected-error@+1{{initalizer for count with side effects is not yet supported}}
  struct CountedByData c1 = { g, getlen() };
  // expected-error@+2{{initalizer for '__counted_by' pointer with side effects is not yet supported}}
  // expected-error@+1{{initalizer for count with side effects is not yet supported}}
  struct CountedByData c2 = { getptr(), getlen() };
  // expected-error@+1{{initalizer for '__counted_by' pointer with side effects is not yet supported}}
  struct CountedByData c3 = { getptr(), 0 };
  // expected-error@+1{{initalizer for '__counted_by' pointer with side effects is not yet supported}}
  struct CountedByData c4 = { getptr() };
  // expected-error@+1{{initalizer for '__counted_by' pointer with side effects is not yet supported}}
  struct CountedByData c5 = { .ptr = getptr() };
  // expected-error@+1{{initalizer for count with side effects is not yet supported}}
  struct CountedByData c6 = { .len = getlen() };
  // expected-error@+2{{initalizer for '__counted_by' pointer with side effects is not yet supported}}
  // expected-error@+1{{initalizer for count with side effects is not yet supported}}
  struct CountedByData c7 = { .ptr = getptr(), .len = getlen() };
  // expected-error@+1{{initalizer for count with side effects is not yet supported}}
  struct CountedBySizedByDataMix c8 = { g, g, getlen() };

  int* ptr;
  struct CountedByDataWithSubStruct c9 = {ptr, 4, {0}}; // OK
  // expected-warning@+1{{initializer 'side_effect()' has a side effect; this may lead to an unexpected result because the evaluation order of initialization list expressions is indeterminate}}
  struct CountedByDataWithSubStruct c10 = {ptr, 4, {side_effect()}};
  struct CountedByDataWithSubStructAtStart c11 = {{0}, ptr, 4}; // OK
  // expected-warning@+1{{initializer 'side_effect()' has a side effect; this may lead to an unexpected result because the evaluation order of initialization list expressions is indeterminate}}
  struct CountedByDataWithSubStructAtStart c12 = {{side_effect()}, ptr, 4};
  struct CountedByDataWithOtherField c13 = {ptr, 4, 0}; // OK
  // expected-warning@+1{{initializer 'side_effect()' has a side effect; this may lead to an unexpected result because the evaluation order of initialization list expressions is indeterminate}}
  struct CountedByDataWithOtherField c14 = {ptr, 4, side_effect()};
}

void TestSizedBy(void) {
  // expected-error@+1{{initalizer for size with side effects is not yet supported}}
  struct SizedByData s1 = { g, getlen() };
  // expected-error@+2{{initalizer for '__sized_by' pointer with side effects is not yet supported}}
  // expected-error@+1{{initalizer for size with side effects is not yet supported}}
  struct SizedByData s2 = { getptr(), getlen() };
  // expected-error@+1{{initalizer for '__sized_by' pointer with side effects is not yet supported}}
  struct SizedByData s3 = { getptr(), 0 };
  // expected-warning@+2{{possibly initializing 's4.ptr' of type 'void *__single __sized_by(len)' (aka 'void *__single') and implicit size value of 0 with non-null, which creates a non-dereferenceable pointer; explicitly set size value to 0 to remove this warning}}
  // expected-error@+1{{initalizer for '__sized_by' pointer with side effects is not yet supported}}
  struct SizedByData s4 = { getptr() };
  // expected-warning@+2{{possibly initializing 's5.ptr' of type 'void *__single __sized_by(len)' (aka 'void *__single') and implicit size value of 0 with non-null, which creates a non-dereferenceable pointer; explicitly set size value to 0 to remove this warning}}
  // expected-error@+1{{initalizer for '__sized_by' pointer with side effects is not yet supported}}
  struct SizedByData s5 = { .ptr = getptr() };
  // expected-error@+1{{initalizer for size with side effects is not yet supported}}
  struct SizedByData s6 = { .len = getlen() };
  // expected-error@+2{{initalizer for '__sized_by' pointer with side effects is not yet supported}}
  // expected-error@+1{{initalizer for size with side effects is not yet supported}}
  struct SizedByData s7 = { .ptr = getptr(), .len = getlen() };
}

void TestCountedByOrNull(void) {
  // expected-error@+1{{initalizer for count with side effects is not yet supported}}
  struct CountedByOrNullData c1 = { g, getlen() };
  // expected-error@+2{{initalizer for '__counted_by_or_null' pointer with side effects is not yet supported}}
  // expected-error@+1{{initalizer for count with side effects is not yet supported}}
  struct CountedByOrNullData c2 = { getptr(), getlen() };
  // expected-error@+1{{initalizer for '__counted_by_or_null' pointer with side effects is not yet supported}}
  struct CountedByOrNullData c3 = { getptr(), 0 };
  // expected-error@+1{{initalizer for '__counted_by_or_null' pointer with side effects is not yet supported}}
  struct CountedByOrNullData c4 = { getptr() };
  // expected-error@+1{{initalizer for '__counted_by_or_null' pointer with side effects is not yet supported}}
  struct CountedByOrNullData c5 = { .ptr = getptr() };
  // expected-error@+1{{initalizer for count with side effects is not yet supported}}
  struct CountedByOrNullData c6 = { .len = getlen() };
  // expected-error@+2{{initalizer for '__counted_by_or_null' pointer with side effects is not yet supported}}
  // expected-error@+1{{initalizer for count with side effects is not yet supported}}
  struct CountedByOrNullData c7 = { .ptr = getptr(), .len = getlen() };
}

void TestSizedByOrNull(void) {
  // expected-error@+1{{initalizer for size with side effects is not yet supported}}
  struct SizedByOrNullData s1 = { g, getlen() };
  // expected-error@+2{{initalizer for '__sized_by_or_null' pointer with side effects is not yet supported}}
  // expected-error@+1{{initalizer for size with side effects is not yet supported}}
  struct SizedByOrNullData s2 = { getptr(), getlen() };
  // expected-error@+1{{initalizer for '__sized_by_or_null' pointer with side effects is not yet supported}}
  struct SizedByOrNullData s3 = { getptr(), 0 };
  // expected-warning@+2{{possibly initializing 's4.ptr' of type 'void *__single __sized_by_or_null(len)' (aka 'void *__single') and implicit size value of 0 with non-null, which creates a non-dereferenceable pointer; explicitly set size value to 0 to remove this warning}}
  // expected-error@+1{{initalizer for '__sized_by_or_null' pointer with side effects is not yet supported}}
  struct SizedByOrNullData s4 = { getptr() };
  // expected-warning@+2{{possibly initializing 's5.ptr' of type 'void *__single __sized_by_or_null(len)' (aka 'void *__single') and implicit size value of 0 with non-null, which creates a non-dereferenceable pointer; explicitly set size value to 0 to remove this warning}}
  // expected-error@+1{{initalizer for '__sized_by_or_null' pointer with side effects is not yet supported}}
  struct SizedByOrNullData s5 = { .ptr = getptr() };
  // expected-error@+1{{initalizer for size with side effects is not yet supported}}
  struct SizedByOrNullData s6 = { .len = getlen() };
  // expected-error@+2{{initalizer for '__sized_by_or_null' pointer with side effects is not yet supported}}
  // expected-error@+1{{initalizer for size with side effects is not yet supported}}
  struct SizedByOrNullData s7 = { .ptr = getptr(), .len = getlen() };
}


void TestRanged(void) {
  // expected-error@+1{{initalizer for '__ended_by' pointer with side effects is not yet supported}}
  struct RangedData r1 = { getptr(), g, g };
  // expected-error@+1{{initalizer for '__ended_by' pointer with side effects is not yet supported}}
  struct RangedData r2 = { g, getptr(), g };
  // expected-error@+1{{initalizer for end pointer with side effects is not yet supported}}
  struct RangedData r3 = { g, g, getptr() };
  // expected-error@+1{{initalizer for '__ended_by' pointer with side effects is not yet supported}}
  struct RangedData r4 = { .start = getptr(), .iter = g, .end = g };
  // expected-error@+1{{initalizer for '__ended_by' pointer with side effects is not yet supported}}
  struct RangedData r5 = { .start = g, .iter = getptr(), .end = g };
  // expected-error@+1{{initalizer for end pointer with side effects is not yet supported}}
  struct RangedData r6 = { .start = g, .iter = g, .end = getptr() };
  // expected-error@+3{{initalizer for end pointer with side effects is not yet supported}}
  // expected-error@+2{{initalizer for '__ended_by' pointer with side effects is not yet supported}}
  // expected-error@+1{{initalizer for '__ended_by' pointer with side effects is not yet supported}}
  struct RangedData r7 = { getptr(), getptr(), getptr() };
}
