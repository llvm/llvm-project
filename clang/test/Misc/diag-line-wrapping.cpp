// RUN: %clang_cc1 -fsyntax-only -fmessage-length 60 %s 2>&1 | FileCheck %s

struct B { void f(); };
struct D1 : B {};
struct D2 : B {};
struct DD : D1, D2 {
  void g() { f(); }
  // Ensure that after line-wrapping takes place, we preserve artificial
  // newlines introduced to manually format a section of the diagnostic text.
  // CHECK: {{.*}}: error:
  // CHECK: struct DD -> struct D1 -> struct B
  // CHECK: struct DD -> struct D2 -> struct B
}
