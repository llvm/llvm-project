// Test macro preservation in C++20 Header Units.

// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
// RUN: cd %t

// RUN: %clang_cc1 -std=c++20 -emit-header-unit -xc++-user-header hu-01.h \
// RUN: -o hu-01.pcm

// RUN: %clang_cc1 -std=c++20 -module-file-info hu-01.pcm | \
// RUN: FileCheck --check-prefix=CHECK-HU %s -DTDIR=%t

// RUN: %clang_cc1 -std=c++20 -emit-header-unit -xc++-user-header hu-02.h \
// RUN: -DFOO -fmodule-file=hu-01.pcm -o hu-02.pcm  -Rmodule-import 2>&1 | \
// RUN: FileCheck --check-prefix=CHECK-IMP %s -DTDIR=%t

// RUN: %clang_cc1 -std=c++20 -module-file-info hu-02.pcm | \
// RUN: FileCheck --check-prefix=CHECK-HU2 %s -DTDIR=%t

// RUN: %clang_cc1 -std=c++20 -emit-module-interface importer-01.cpp \
// RUN:  -fmodule-file=hu-02.pcm -o B.pcm -verify

// RUN: %clang_cc1 -std=c++20 -emit-module-interface importer-02.cpp \
// RUN:  -fmodule-file=hu-02.pcm -o C.pcm -Rmodule-import 2>&1 | \
// RUN:  FileCheck --check-prefix=CHECK-IMP-HU2 %s -DTDIR=%t

//--- hu-01.h
#ifndef __GUARD
#define __GUARD

int baz(int);
#define FORTYTWO 42

#define SHOULD_NOT_BE_DEFINED -1
#undef SHOULD_NOT_BE_DEFINED

#endif // __GUARD
// expected-no-diagnostics

// CHECK-HU:  ====== C++20 Module structure ======
// CHECK-HU-NEXT:  Header Unit './hu-01.h' is the Primary Module at index #1

//--- hu-02.h
export import "hu-01.h"; // expected-warning {{the implementation of header units is in an experimental phase}}
#if !defined(FORTYTWO) || FORTYTWO != 42
#error FORTYTWO missing in hu-02
#endif

#ifndef __GUARD
#error __GUARD missing in hu-02
#endif

#ifdef SHOULD_NOT_BE_DEFINED
#error SHOULD_NOT_BE_DEFINED is visible
#endif

#define KAP 6174

#ifdef FOO
#define FOO_BRANCH(X) (X) + 1
inline int foo(int x) {
  if (x == FORTYTWO)
    return FOO_BRANCH(x);
  return FORTYTWO;
}
#else
#define BAR_BRANCH(X) (X) + 2
inline int bar(int x) {
  if (x == FORTYTWO)
    return BAR_BRANCH(x);
  return FORTYTWO;
}
#endif

// CHECK-IMP: remark: importing module './hu-01.h' from 'hu-01.pcm'
// CHECK-HU2:  ====== C++20 Module structure ======
// CHECK-HU2-NEXT:  Header Unit './hu-02.h' is the Primary Module at index #2
// CHECK-HU2-NEXT:   Exports:
// CHECK-HU2-NEXT:    Header Unit './hu-01.h' is at index #1
// expected-no-diagnostics

//--- importer-01.cpp
export module B;
import "hu-02.h"; // expected-warning {{the implementation of header units is in an experimental phase}}

int success(int x) {
  return foo(FORTYTWO + x + KAP);
}

int fail(int x) {
  return bar(FORTYTWO + x + KAP); // expected-error {{use of undeclared identifier 'bar'}}
  // expected-note@* {{'baz' declared here}}
}

//--- importer-02.cpp
export module C;
import "hu-02.h"; // expected-warning {{the implementation of header units is in an experimental phase}}

int success(int x) {
  return foo(FORTYTWO + x + KAP);
}

// CHECK-IMP-HU2: remark: importing module './hu-02.h' from 'hu-02.pcm'
// CHECK-IMP-HU2: remark: importing module './hu-01.h' into './hu-02.h' from '[[TDIR]]{{[/\\]}}hu-01.pcm'
