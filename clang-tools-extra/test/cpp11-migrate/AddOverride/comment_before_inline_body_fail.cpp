// RUN: grep -Ev "// *[A-Z-]+:" %s > %t.cpp
// RUN: cpp11-migrate -add-override %t.cpp -- -I %S
// RUN: FileCheck -input-file=%t.cpp %s
// XFAIL: *

class A {
public:
  virtual void h() const;
  // CHECK: virtual void h() const;
};

// Test that the override is correctly placed if there
// is an inline comment between the function declaration
// and the function body.
// This test fails with the override keyword being added
// to the end of the comment. This failure occurs because
// the insertion point is incorrectly calculated if there
// is an inline comment before the method body.
class B : public A {
public:
  virtual void h() const // comment
  { }
  // CHECK: virtual void h() const override
};

