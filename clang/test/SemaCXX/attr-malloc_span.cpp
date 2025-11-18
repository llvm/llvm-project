// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %s

struct span_with_static {
  void *ptr;
  int n;
  static int static_field;
};

span_with_static  returns_span_with_static  (void) __attribute((malloc_span)); // no-warning

class SomeClass {
public:
  int Data;
};

// Returning pointers to data members is not allowed.
struct DataMemberSpan {
  int SomeClass::* member_ptr;
  int n;
};

// expected-warning@+2 {{attribute only applies to functions that return span-like structures}}
// expected-note@+1 {{returned struct fields are not a supported combination}}
DataMemberSpan returns_data_member_span(void) __attribute((malloc_span)) {
  return DataMemberSpan{};
}

// Returning pointers to member functions is not allowed.
struct MemberFuncSpan {
  void (SomeClass::*member_func_ptr)();
  int n;
};

// expected-warning@+2 {{attribute only applies to functions that return span-like structures}}
// expected-note@+1 {{returned struct fields are not a supported combination}}
MemberFuncSpan returns_member_func_span(void) __attribute((malloc_span)) {
  return MemberFuncSpan{};
}

template<typename FirstType, typename SecondType>
struct Pair {
  FirstType first;
  SecondType second;
};

Pair<int*, int> returns_templated_span1(void) __attribute((malloc_span)) { // no-warning
  return Pair<int*, int>{};
}

Pair<int*, int*> returns_templated_span2(void) __attribute((malloc_span)) { // no-warning
  return Pair<int*, int*>{};
}

// expected-warning@+2 {{attribute only applies to functions that return span-like structures}}
// expected-note@+1 {{returned struct fields are not a supported combination for a span-like type}}
Pair<int, int> returns_templated_span3(void) __attribute((malloc_span)) {
  return Pair<int, int>{};
}

// Verify that semantic checks are done on dependent types.

struct GoodSpan {
  void *ptr;
  int n;
};

struct BadSpan {
  int n;
};

template <typename T>
// expected-warning@+2 {{'malloc_span' attribute only applies to functions that return span-like structures}}
// expected-note@+1 {{returned struct has 1 fields, expected 2}}
T produce_span() __attribute((malloc_span)) {
  return T{};
}

void TestGoodBadSpan() {
  produce_span<GoodSpan>(); // no-warnings
  // expected-note@+1 {{in instantiation of function template specialization 'produce_span<BadSpan>' requested here}}
  produce_span<BadSpan>();
}

// Ensure that trailing return types are also supported.
__attribute__((malloc_span)) auto trailing_return_type(int size)  -> GoodSpan { // no-warning
  return GoodSpan{};
}

template<typename T>
// expected-warning@+2 {{'malloc_span' attribute only applies to functions that return span-like structures}}
// expected-note@+1 {{returned struct has 1 fields, expected 2}}
__attribute__((malloc_span)) auto templated_trailing_return_type()  -> T {
  return T{};
}

void TestGoodBadTrailingReturnType() {
  templated_trailing_return_type<GoodSpan>(); // no-warnings
  // expected-note@+1 {{in instantiation of function template specialization 'templated_trailing_return_type<BadSpan>' requested here}}
  templated_trailing_return_type<BadSpan>();
}

__attribute((malloc_span)) auto trailing_return_temmplate_good(void) -> Pair<int*, int> { // no-warning
  return Pair<int*, int>{};
}

// expected-warning@+2 {{attribute only applies to functions that return span-like structures}}
// expected-note@+1 {{returned struct fields are not a supported combination for a span-like type}}
__attribute((malloc_span)) auto trailing_return_temmplate_bad(void) -> Pair<int, int> {
  return Pair<int, int>{};
}

struct Base {
  void *other_p;
};

struct ChildSpan : Base {
  void *p;
  int n;
};

// expected-warning@+2 {{attribute only applies to functions that return span-like structures}}
// expected-note@+1 {{returned type inherits from a base class}}
__attribute((malloc_span)) ChildSpan return_child_span(void);
