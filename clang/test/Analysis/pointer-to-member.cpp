// RUN: %clang_cc1 -analyze -analyzer-checker=core,debug.ExprInspection -analyzer-ipa=inlining -verify %s

void clang_analyzer_eval(bool);

struct A {
  // This conversion operator allows implicit conversion to bool but not to other integer types.
  typedef A * (A::*MemberPointer);
  operator MemberPointer() const { return m_ptr ? &A::m_ptr : 0; }

  A *m_ptr;
};

void testConditionalUse() {
  A obj;

  obj.m_ptr = &obj;
  clang_analyzer_eval(obj.m_ptr); // expected-warning{{TRUE}}
  clang_analyzer_eval(&A::m_ptr); // expected-warning{{TRUE}}
  clang_analyzer_eval(obj); // expected-warning{{TRUE}}

  obj.m_ptr = 0;
  clang_analyzer_eval(obj.m_ptr); // expected-warning{{FALSE}}
  clang_analyzer_eval(A::MemberPointer(0)); // expected-warning{{FALSE}}
  clang_analyzer_eval(obj); // expected-warning{{FALSE}}
}

// ---------------
// FALSE NEGATIVES
// ---------------

bool testDereferencing() {
  A obj;
  obj.m_ptr = 0;

  A::MemberPointer member = &A::m_ptr;

  // FIXME: Should be TRUE.
  clang_analyzer_eval(obj.*member == 0); // expected-warning{{UNKNOWN}}

  member = 0;

  // FIXME: Should emit a null dereference.
  return obj.*member; // no-warning
}
