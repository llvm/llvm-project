// Check handling definitions from a file that is accessed both as non-modular and modular.

// RUN: rm -rf %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -fsyntax-only -I %t/headers-c -I %t/headers-c/sub %t/test.c -verify
// RUN: %clang_cc1 -fsyntax-only -I %t/headers-c -I %t/headers-c/sub %t/test.c -verify \
// RUN:   -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/modules.cache

// RUN: %clang_cc1 -fsyntax-only -I %t/headers-objc -I %t/headers-objc/sub %t/test.m -verify
// RUN: %clang_cc1 -fsyntax-only -I %t/headers-objc -I %t/headers-objc/sub %t/test.m -verify \
// RUN:   -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/modules.cache

// RUN: %clang_cc1 -fsyntax-only -I %t/headers-cxx -I %t/headers-cxx/sub %t/test.cpp -verify
// RUN: %clang_cc1 -fsyntax-only -I %t/headers-cxx -I %t/headers-cxx/sub %t/test.cpp -verify \
// RUN:   -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/modules.cache

// RUN: %clang_cc1 -fsyntax-only -I %t/headers-unguarded -I %t/headers-unguarded/sub %t/unguarded.c -verify
// RUN: %clang_cc1 -fsyntax-only -I %t/headers-unguarded -I %t/headers-unguarded/sub %t/unguarded.c -verify \
// RUN:   -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/modules.cache

// RUN: %clang_cc1 -fsyntax-only -I %t/headers-mismatched -I %t/headers-mismatched/sub %t/mismatched.c -verify \
// RUN:   -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/modules.cache

//--- headers-c/top.h
#ifndef TOP_H
#define TOP_H

#include <sub/sub.h>

#endif

//--- headers-c/sub/sub.h
#ifndef SUB_H
#define SUB_H

#include <top.h>
typedef int TestTypedef;

struct TestStruct {
  int a;
  int b: 3;
};

union TestUnion {
  int x;
  float y;
};

struct WithAnonymous {
  struct {
    char p;
  };
  struct {
    int z;
  } nested;
};

#define CUSTOM_STRUCT(name) struct name##Struct

CUSTOM_STRUCT(MacroBased) {
  int m;
};

enum Seasons {
  kSeasonWinter = 0,
  kSeasonSpring,
};

inline int square(int x) {
  return x * x;
}

#endif

//--- headers-c/module.modulemap
module top_c {
  header "top.h"
  export *
}

//--- test.c
// expected-no-diagnostics
// Access 'sub/sub.h' in non-modular way.
#include <sub.h>


//--- headers-objc/top.h
#import <sub/sub.h>

//--- headers-objc/sub/sub.h
#import <top.h>

@protocol TestProto
- (void)testProtocolMethod;
@end

__attribute__((objc_root_class))
@interface TestClass {
  int _a;
}
- (void)testMethod:(float)b;
@end

//--- headers-objc/module.modulemap
module top_objc {
  header "top.h"
  export *
}

//--- test.m
// expected-no-diagnostics
// Access 'sub/sub.h' in non-modular way.
#import <sub.h>


//--- headers-cxx/top.h
#pragma once
#import <sub/sub.h>

//--- headers-cxx/sub/sub.h
#pragma once
#include <top.h>

inline int GlobalVar = 3;

template <class T> class GenericClass {
  T field;
};

template <bool B, class T, class F> struct condition { using type = F; };
template <class T, class F> struct condition<true, T, F> { using type = T; };

template <typename T> void printGeneric(const T &val) {
  // empty
}

template <> void printGeneric<int>(const int &val) {
  // still empty
}

//--- headers-cxx/module.modulemap
module top_cxx {
  header "top.h"
  export *
}

//--- test.cpp
// expected-no-diagnostics
// Access 'sub/sub.h' in non-modular way.
#import <sub.h>


//--- headers-unguarded/top.h
#ifndef TOP_H
#define TOP_H
#include <sub/sub.h>
#endif

//--- headers-unguarded/sub/sub.h
#include <top.h>

struct UnguardedStruct {
  float x;
  int y;
};

//--- headers-unguarded/module.modulemap
module top_unguarded {
  header "top.h"
  export *
}

//--- unguarded.c
// Access 'sub/sub.h' in non-modular way.
#include <sub.h>
// expected-error@sub.h:* {{redefinition of 'UnguardedStruct'}}
// expected-note@top.h:* {{sub.h' included multiple times, additional include site}}
// expected-note@unguarded.c:* {{sub.h' included multiple times, additional include site}}
// expected-note@sub.h:* {{unguarded header; consider using #ifdef guards or #pragma once}}
#if __has_feature(modules)
// expected-note@module.modulemap:* {{top_unguarded defined here}}
#endif


//--- headers-mismatched/top.h
#ifndef TOP_H
#define TOP_H
#include <sub/sub.h>
#endif

//--- headers-mismatched/sub/sub.h
#ifndef SUB_H
#define SUB_H

#include <top.h>

struct MismatchedStruct {
  int x;
#ifdef EXTRA_FIELD
  char z;
#endif
};

#endif

//--- headers-mismatched/module.modulemap
module top_mismatched {
  header "top.h"
  export *
}

//--- mismatched.c
#define EXTRA_FIELD 1
#include <sub.h>
// expected-error@sub.h:* {{type 'struct MismatchedStruct' has incompatible definitions}}
// expected-note@sub.h:* {{field 'z' has type 'char' here}}
// expected-note@sub.h:* {{no corresponding field here}}
