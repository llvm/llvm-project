// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %s
// expected-no-diagnostics

// Ensure the following out of line friend declaration doesn't cause the compiler to crash.

namespace GH120857
{

class A {
  friend bool operator==(const A&, const A&);
  friend class B;
};

bool operator==(const A&, const A&) = default;

}

