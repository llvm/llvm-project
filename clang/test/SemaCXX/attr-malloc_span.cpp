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

DataMemberSpan returns_data_member_span(void) __attribute((malloc_span)) { // expected-warning {{attribute only applies to functions that return span-like structures}}
  return DataMemberSpan{};
}

// Returning pointers to member functions is not allowed.
struct MemberFuncSpan {
  void (SomeClass::*member_func_ptr)();
  int n;
};

MemberFuncSpan returns_member_func_span(void) __attribute((malloc_span)) { // expected-warning {{attribute only applies to functions that return span-like structures}}
  return MemberFuncSpan{};
}

