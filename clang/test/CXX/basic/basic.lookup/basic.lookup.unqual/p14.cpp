// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

// C++0x [basic.lookup.unqual]p14:
//   If a variable member of a namespace is defined outside of the
//   scope of its namespace then any name used in the definition of
//   the variable member (after the declarator-id) is looked up as if
//   the definition of the variable member occurred in its namespace.

namespace N { 
  struct S {};
  S i; 
  extern S j;
  extern S j2;
} 

int i = 2; 
N::S N::j = i;
N::S N::j2(i);
