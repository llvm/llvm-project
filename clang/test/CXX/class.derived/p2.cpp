// RUN: %clang_cc1 %s -fsyntax-only -verify
// expected-no-diagnostics

// "During the lookup for a base class name, non-type names are ignored"
namespace PR5840 {
  struct Base {};
  int Base = 10;
  struct Derived : Base {};
} // namespace PR5840

namespace issue_16855 {
  struct x {};
  namespace
  {
      namespace x
      {
          struct y : x
          {};
      } // namespace x
  }
} // namespace issue_16855
