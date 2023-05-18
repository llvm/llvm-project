// RUN: %clang_cc1 -fsyntax-only -verify %s

void f5 (int z) { 
  if (z) 
    default:  // expected-error {{not in switch statement}}
      ;
} 

