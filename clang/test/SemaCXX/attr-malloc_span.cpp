// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %s

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
// expected-note@+1 {{span-like type must have a pointer and an integer field or two pointer fields}}
DataMemberSpan returns_data_member_span(void) __attribute((malloc_span)) {
  return DataMemberSpan{};
}

// Returning pointers to member functions is not allowed.
struct MemberFuncSpan {
  void (SomeClass::*member_func_ptr)();
  int n;
};

// expected-warning@+2 {{attribute only applies to functions that return span-like structures}}
// expected-note@+1 {{span-like type must have a pointer and an integer field or two pointer fields}}
MemberFuncSpan returns_member_func_span(void) __attribute((malloc_span)) {
  return MemberFuncSpan{};
}

