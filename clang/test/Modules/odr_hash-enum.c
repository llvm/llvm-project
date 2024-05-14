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
// RUN: %clang_cc1 -fsyntax-only -x c %t/Inputs/first.h
// RUN: %clang_cc1 -fsyntax-only -x c %t/Inputs/second.h

// Build module map file
// RUN: echo "module FirstModule {"     >> %t/Inputs/module.modulemap
// RUN: echo "    header \"first.h\""   >> %t/Inputs/module.modulemap
// RUN: echo "}"                        >> %t/Inputs/module.modulemap
// RUN: echo "module SecondModule {"    >> %t/Inputs/module.modulemap
// RUN: echo "    header \"second.h\""  >> %t/Inputs/module.modulemap
// RUN: echo "}"                        >> %t/Inputs/module.modulemap

// Run test
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/cache -x c -I%t/Inputs -verify %s

#if !defined(FIRST) && !defined(SECOND)
#include "first.h"
#include "second.h"
#endif

#if defined(FIRST)
enum DifferentEnumConstants { kDifferentEnumConstantsValueFirst };
#elif defined(SECOND)
enum DifferentEnumConstants { kDifferentEnumConstantsValueSecond };
#else
enum DifferentEnumConstants differentEnumConstants;
// expected-error@second.h:* {{'kDifferentEnumConstantsValueSecond' from module 'SecondModule' is not present in definition of 'enum DifferentEnumConstants' in module 'FirstModule'}}
// expected-note@first.h:* {{definition has no member 'kDifferentEnumConstantsValueSecond'}}
#endif

#if defined(FIRST)
enum DifferentEnumValues { kDifferentEnumValue = 0 };
#elif defined(SECOND)
enum DifferentEnumValues { kDifferentEnumValue = 1 };
#else
enum DifferentEnumValues differentEnumValue;
// expected-error@first.h:* {{'DifferentEnumValues' has different definitions in different modules; definition in module 'FirstModule' first difference is 1st element 'kDifferentEnumValue' has an initializer}}
// expected-note@second.h:* {{but in 'SecondModule' found 1st element 'kDifferentEnumValue' has different initializer}}
#endif

#if defined(FIRST)
enum {
    kAnonymousEnumValueFirst = 1,
};
#elif defined(SECOND)
enum {
    kAnonymousEnumValueSecond = 2,
};
#else
// Anonymous enums don't have to match, no errors expected.
int anonymousEnumValue = kAnonymousEnumValueFirst + kAnonymousEnumValueSecond;
#endif

// Keep macros contained to one file.
#ifdef FIRST
#undef FIRST
#endif

#ifdef SECOND
#undef SECOND
#endif
