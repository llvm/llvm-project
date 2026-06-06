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
// RUN: %clang_cc1 -fsyntax-only -x c++ -std=c++11 -fblocks %t/Inputs/first.h
// RUN: %clang_cc1 -fsyntax-only -x c++ -std=c++11 -fblocks %t/Inputs/second.h

// Build module map file
// RUN: echo "module FirstModule {"     >> %t/Inputs/module.modulemap
// RUN: echo "    header \"first.h\""   >> %t/Inputs/module.modulemap
// RUN: echo "}"                        >> %t/Inputs/module.modulemap
// RUN: echo "module SecondModule {"    >> %t/Inputs/module.modulemap
// RUN: echo "    header \"second.h\""  >> %t/Inputs/module.modulemap
// RUN: echo "}"                        >> %t/Inputs/module.modulemap

// Run test
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps \
// RUN:            -fmodules-cache-path=%t/cache -x c++ -I%t/Inputs \
// RUN:            -verify %s -std=c++11 -fblocks

#if !defined(FIRST) && !defined(SECOND)
#include "first.h"
#include "second.h"
#endif

// Used for testing
#if defined(FIRST)
#define ACCESS public:
#elif defined(SECOND)
#define ACCESS private:
#endif

// TODO: S1 and S2 should generate errors.
namespace Blocks {
#if defined(FIRST)
struct S1 {
  void (^block)(int x) = ^(int x) { };
};
#elif defined(SECOND)
struct S1 {
  void (^block)(int x) = ^(int y) { };
};
#else
S1 s1;
#endif

#if defined(FIRST)
struct S2 {
  int (^block)(int x) = ^(int x) { return x + 1; };
};
#elif defined(SECOND)
struct S2 {
  int (^block)(int x) = ^(int x) { return x; };
};
#else
S2 s2;
#endif

#if defined(FIRST)
struct S3 {
  void run(int (^block)(int x));
};
#elif defined(SECOND)
struct S3 {
  void run(int (^block)(int x, int y));
};
#else
S3 s3;
// expected-error@first.h:* {{'Blocks::S3::run' from module 'FirstModule' is not present in definition of 'Blocks::S3' in module 'SecondModule'}}
// expected-note@second.h:* {{declaration of 'run' does not match}}
#endif

#define DECLS                                       \
  int (^block)(int x) = ^(int x) { return x + x; }; \
  void run(int (^block)(int x, int y));

#if defined(FIRST) || defined(SECOND)
struct Valid1 {
  DECLS
};
#else
Valid1 v1;
#endif

#if defined(FIRST) || defined(SECOND)
struct Invalid1 {
  DECLS
  ACCESS
};
#else
Invalid1 i1;
// expected-error@second.h:* {{'Blocks::Invalid1' has different definitions in different modules; first difference is definition in module 'SecondModule' found private access specifier}}
// expected-note@first.h:* {{but in 'FirstModule' found public access specifier}}
#endif

#undef DECLS
}

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
