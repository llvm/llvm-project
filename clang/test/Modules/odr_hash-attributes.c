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
// RUN: %clang_cc1 -fsyntax-only -x c %t/Inputs/first.h -fblocks -fobjc-arc -Wno-objc-root-class
// RUN: %clang_cc1 -fsyntax-only -x c %t/Inputs/second.h -fblocks -fobjc-arc -Wno-objc-root-class

// Build module map file
// RUN: echo "module FirstModule {"     >> %t/Inputs/module.map
// RUN: echo "    header \"first.h\""   >> %t/Inputs/module.map
// RUN: echo "}"                        >> %t/Inputs/module.map
// RUN: echo "module SecondModule {"    >> %t/Inputs/module.map
// RUN: echo "    header \"second.h\""  >> %t/Inputs/module.map
// RUN: echo "}"                        >> %t/Inputs/module.map

// Run test
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/cache -x c -I%t/Inputs -verify %s -fblocks -fobjc-arc -Wno-objc-root-class -fodr-hash-attributes

#if !defined(FIRST) && !defined(SECOND)
#include "first.h"
#include "second.h"
#endif

#if defined(FIRST) || defined(SECOND)
#endif

#if defined(FIRST)
typedef struct __attribute__((objc_bridge_mutable(NSMutableDictionary))) __CFDictionary *CFMutableDictionaryRef;
typedef struct __CFArray *CFMutableArrayRef;
typedef struct __attribute__ ((objc_bridge(NSError))) __CFErrorRef *CFErrorRef;
typedef struct __attribute__ ((objc_bridge(NSSomething), objc_bridge_mutable(NSMutableSomething)))  __CFSomethingRef *CFSomethingRef;
#elif defined(SECOND)
typedef struct __CFDictionary *CFMutableDictionaryRef;
typedef struct __attribute__((objc_bridge_mutable(NSMutableArray))) __CFArray *CFMutableArrayRef;
typedef struct __attribute__ ((objc_bridge(NSOtherError))) __CFErrorRef *CFErrorRef;
// This shouldn't trigger ODR -> order does not matter.
typedef struct __attribute__ ((objc_bridge_mutable(NSMutableSomething), objc_bridge(NSSomething)))  __CFSomethingRef *CFSomethingRef;
#else
CFMutableDictionaryRef D;
CFMutableArrayRef A;
CFErrorRef E;
CFSomethingRef S;
// expected-error@first.h:* {{'__CFDictionary' has different definitions in different modules; first difference is definition in module 'FirstModule' found  __attribute__((objc_bridge_mutable(NSMutableDictionary)))}}
// expected-note@second.h:* {{but in 'SecondModule' found no attribute}}
// expected-error@first.h:* {{'__CFArray' has different definitions in different modules; first difference is definition in module 'FirstModule' found no attribute}}
// expected-note@second.h:* {{but in 'SecondModule' found  __attribute__((objc_bridge_mutable(NSMutableArray)))}}
// expected-error@first.h:* {{'__CFErrorRef' has different definitions in different modules; first difference is definition in module 'FirstModule' found  __attribute__((objc_bridge(NSError)))}}
// expected-note@second.h:* {{but in 'SecondModule' found  __attribute__((objc_bridge(NSOtherError)))}}
#endif

// Keep macros contained to one file.
#ifdef FIRST
#undef FIRST
#endif

#ifdef SECOND
#undef SECOND
#endif
