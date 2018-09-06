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
// RUN: %clang_cc1 -fsyntax-only -x c++ -std=c++11 %t/Inputs/first.h -fzvector
// RUN: %clang_cc1 -fsyntax-only -x c++ -std=c++11 %t/Inputs/second.h -fzvector

// Build module map file
// RUN: echo "module FirstModule {"     >> %t/Inputs/module.map
// RUN: echo "    header \"first.h\""   >> %t/Inputs/module.map
// RUN: echo "}"                        >> %t/Inputs/module.map
// RUN: echo "module SecondModule {"    >> %t/Inputs/module.map
// RUN: echo "    header \"second.h\""  >> %t/Inputs/module.map
// RUN: echo "}"                        >> %t/Inputs/module.map

// Run test
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/cache -x c++ -I%t/Inputs -verify %s -std=c++11 -fzvector

#if !defined(FIRST) && !defined(SECOND)
#include "first.h"
#include "second.h"
#endif

namespace Types {
namespace Vector {
#if defined(FIRST)
struct Invalid1 {
  __attribute((vector_size(8))) int x;
};
struct Invalid2 {
  __attribute((vector_size(8))) int x;
};
struct Invalid3 {
  __attribute((vector_size(16))) int x;
};
struct Valid {
  __attribute((vector_size(8))) int x1;
  __attribute((vector_size(16))) int x2;
  __attribute((vector_size(8))) unsigned x3;
  __attribute((vector_size(16))) long x4;
  vector unsigned x5;
  vector int x6;
};
#elif defined(SECOND)
struct Invalid1 {
  __attribute((vector_size(16))) int x;
};
struct Invalid2 {
  __attribute((vector_size(8))) unsigned x;
};
struct Invalid3 {
  vector unsigned x;
};
struct Valid {
  __attribute((vector_size(8))) int x1;
  __attribute((vector_size(16))) int x2;
  __attribute((vector_size(8))) unsigned x3;
  __attribute((vector_size(16))) long x4;
  vector unsigned x5;
  vector int x6;
};
#else
Invalid1 i1;
// expected-error@second.h:* {{'Types::Vector::Invalid1::x' from module 'SecondModule' is not present in definition of 'Types::Vector::Invalid1' in module 'FirstModule'}}
// expected-note@first.h:* {{declaration of 'x' does not match}}
Invalid2 i2;
// expected-error@second.h:* {{'Types::Vector::Invalid2::x' from module 'SecondModule' is not present in definition of 'Types::Vector::Invalid2' in module 'FirstModule'}}
// expected-note@first.h:* {{declaration of 'x' does not match}}
Invalid3 i3;
// expected-error@second.h:* {{'Types::Vector::Invalid3::x' from module 'SecondModule' is not present in definition of 'Types::Vector::Invalid3' in module 'FirstModule'}}
// expected-note@first.h:* {{declaration of 'x' does not match}}

Valid v;
#endif
}  // namespace Vector



namespace ExtVector {
}  // namespace ExtVector
#if defined(FIRST)
struct Invalid {
  using f = __attribute__((ext_vector_type(4))) float;
};
struct Valid {
  using f = __attribute__((ext_vector_type(8))) float;
};
#elif defined(SECOND)
struct Invalid {
  using f = __attribute__((ext_vector_type(8))) float;
};
struct Valid {
  using f = __attribute__((ext_vector_type(8))) float;
};
#else
Invalid i;
// expected-error@first.h:* {{'Types::Invalid::f' from module 'FirstModule' is not present in definition of 'Types::Invalid' in module 'SecondModule'}}
// expected-note@second.h:* {{declaration of 'f' does not match}}

Valid v;
#endif

}  // namespace Types


// Keep macros contained to one file.
#ifdef FIRST
#undef FIRST
#endif

#ifdef SECOND
#undef SECOND
#endif

#ifdef ACCESS
#undef ACCESS
#endif
