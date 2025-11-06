// Clear and create directories
// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: mkdir %t/cache
// RUN: mkdir %t/Inputs

// Build first header file
// RUN: echo "#define FIRST" >> %t/Inputs/first.h
// RUN: cat %s               >> %t/Inputs/first.h

// Build second header file
// RUN: echo "#define SECOND" >> %t/Inputs/second.h
// RUN: cat %s                >> %t/Inputs/second.h

// Test that each header can compile
// RUN: %clang_cc1 -fsyntax-only -x c++ %t/Inputs/first.h -cl-std=CL2.0
// RUN: %clang_cc1 -fsyntax-only -x c++ %t/Inputs/second.h -cl-std=CL2.0

// Build module map file
// RUN: echo "module FirstModule {"     >> %t/Inputs/module.modulemap
// RUN: echo "    header \"first.h\""   >> %t/Inputs/module.modulemap
// RUN: echo "}"                        >> %t/Inputs/module.modulemap
// RUN: echo "module SecondModule {"    >> %t/Inputs/module.modulemap
// RUN: echo "    header \"second.h\""  >> %t/Inputs/module.modulemap
// RUN: echo "}"                        >> %t/Inputs/module.modulemap

// Run test
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/cache -x c++ -I%t/Inputs -verify %s -cl-std=CL2.0

#if !defined(FIRST) && !defined(SECOND)
#include "first.h"
#include "second.h"
#endif


#if defined(FIRST)
void invalid1() {
  typedef read_only pipe int x;
}
void invalid2() {
  typedef read_only pipe int x;
}
void valid() {
  typedef read_only pipe int x;
  typedef write_only pipe int y;
  typedef read_write pipe int z;
}
#elif defined(SECOND)
void invalid1() {
  typedef write_only pipe int x;
}
void invalid2() {
  typedef read_only pipe float x;
}
void valid() {
  typedef read_only pipe int x;
  typedef write_only pipe int y;
  typedef read_write pipe int z;
}
#else
void run() {
  invalid1();
// expected-error@second.h:* {{'invalid1' has different definitions in different modules; definition in module 'SecondModule' first difference is function body}}
// expected-note@first.h:* {{but in 'FirstModule' found a different body}}
  invalid2();
// expected-error@second.h:* {{'invalid2' has different definitions in different modules; definition in module 'SecondModule' first difference is function body}}
// expected-note@first.h:* {{but in 'FirstModule' found a different body}}
  valid();
}
#endif


// Keep macros contained to one file.
#ifdef FIRST
#undef FIRST
#endif

#ifdef SECOND
#undef SECOND
#endif
