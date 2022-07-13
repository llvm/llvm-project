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
// RUN: %clang_cc1 -fsyntax-only -x c++ -std=gnu++11 %t/Inputs/first.h
// RUN: %clang_cc1 -fsyntax-only -x c++ -std=gnu++11 %t/Inputs/second.h

// Build module map file
// RUN: echo "module FirstModule {"     >> %t/Inputs/module.map
// RUN: echo "    header \"first.h\""   >> %t/Inputs/module.map
// RUN: echo "}"                        >> %t/Inputs/module.map
// RUN: echo "module SecondModule {"    >> %t/Inputs/module.map
// RUN: echo "    header \"second.h\""  >> %t/Inputs/module.map
// RUN: echo "}"                        >> %t/Inputs/module.map

// Run test
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/cache -x c++ -I%t/Inputs -verify %s -std=gnu++11

#if !defined(FIRST) && !defined(SECOND)
#include "first.h"
#include "second.h"
#endif

namespace Types {
namespace TypeOfExpr {
#if defined(FIRST)
struct Invalid1 {
  typeof(1 + 2) x;
};
double global;
struct Invalid2 {
  typeof(global) x;
};
struct Valid {
  typeof(3) x;
  typeof(x) y;
  typeof(Valid*) self;
};
#elif defined(SECOND)
struct Invalid1 {
  typeof(3) x;
};
int global;
struct Invalid2 {
  typeof(global) x;
};
struct Valid {
  typeof(3) x;
  typeof(x) y;
  typeof(Valid*) self;
};
#else
Invalid1 i1;
// expected-error@first.h:* {{'Types::TypeOfExpr::Invalid1' has different definitions in different modules; first difference is definition in module 'FirstModule' found field 'x' with type 'typeof (1 + 2)' (aka 'int')}}
// expected-note@second.h:* {{but in 'SecondModule' found field 'x' with type 'typeof (3)' (aka 'int')}}
Invalid2 i2;
// expected-error@second.h:* {{'Types::TypeOfExpr::Invalid2::x' from module 'SecondModule' is not present in definition of 'Types::TypeOfExpr::Invalid2' in module 'FirstModule'}}
// expected-note@first.h:* {{declaration of 'x' does not match}}
Valid v;
#endif
}  // namespace TypeOfExpr

namespace TypeOf {
#if defined(FIRST)
struct Invalid1 {
  typeof(int) x;
};
struct Invalid2 {
  typeof(int) x;
};
using T = int;
struct Invalid3 {
  typeof(T) x;
};
struct Valid {
  typeof(int) x;
  using T = typeof(double);
  typeof(T) y;
};
#elif defined(SECOND)
struct Invalid1 {
  typeof(double) x;
};
using I = int;
struct Invalid2 {
  typeof(I) x;
};
using T = short;
struct Invalid3 {
  typeof(T) x;
};
struct Valid {
  typeof(int) x;
  using T = typeof(double);
  typeof(T) y;
};
#else
Invalid1 i1;
// expected-error@second.h:* {{'Types::TypeOf::Invalid1::x' from module 'SecondModule' is not present in definition of 'Types::TypeOf::Invalid1' in module 'FirstModule'}}
// expected-note@first.h:* {{declaration of 'x' does not match}}
Invalid2 i2;
// expected-error@first.h:* {{'Types::TypeOf::Invalid2' has different definitions in different modules; first difference is definition in module 'FirstModule' found field 'x' with type 'typeof(int)' (aka 'int')}}
// expected-note@second.h:* {{but in 'SecondModule' found field 'x' with type 'typeof(Types::TypeOf::I)' (aka 'int')}}
Invalid3 i3;
// expected-error@second.h:* {{'Types::TypeOf::Invalid3::x' from module 'SecondModule' is not present in definition of 'Types::TypeOf::Invalid3' in module 'FirstModule'}}
// expected-note@first.h:* {{declaration of 'x' does not match}}
Valid v;
#endif
}  // namespace TypeOf
}  // namespace Types

// Keep macros contained to one file.
#ifdef FIRST
#undef FIRST
#endif

#ifdef SECOND
#undef SECOND
#endif
