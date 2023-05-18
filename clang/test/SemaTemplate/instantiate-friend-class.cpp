// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics
// PR4794

template <class T> class X
{
  friend class Y;
};
X<int> y;

