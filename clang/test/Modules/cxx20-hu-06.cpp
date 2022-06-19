// Test check that consuming -E -fdirectives-only output produces the expected
// header unit.

// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
// RUN: cd %t

// RUN: %clang_cc1 -std=c++20 -E -fdirectives-only -xc++-user-header hu-01.h \
// RUN: -o hu-01.iih

// RUN: %clang_cc1 -std=c++20 -emit-header-unit \
// RUN: -xc++-user-header-cpp-output hu-01.iih -o hu-01.pcm

// RUN: %clang_cc1 -std=c++20 -emit-header-unit -xc++-user-header hu-02.h \
// RUN: -DFOO -fmodule-file=hu-01.pcm -o hu-02.pcm -Rmodule-import 2>&1 | \
// RUN: FileCheck --check-prefix=CHECK-IMP %s -DTDIR=%t

//--- hu-01.h
#ifndef __GUARD
#define __GUARD

int baz(int);
#define FORTYTWO 42

#define SHOULD_NOT_BE_DEFINED -1
#undef SHOULD_NOT_BE_DEFINED

#endif // __GUARD
// expected-no-diagnostics

//--- hu-02.h
export import "hu-01.h";
#if !defined(FORTYTWO) || FORTYTWO != 42
#error FORTYTWO missing in hu-02
#endif

#ifndef __GUARD
#error __GUARD missing in hu-02
#endif

#ifdef SHOULD_NOT_BE_DEFINED
#error SHOULD_NOT_BE_DEFINED is visible
#endif

// Make sure that we have not discarded macros from the builtin file.
#ifndef __cplusplus
#error we dropped a defined macro
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
