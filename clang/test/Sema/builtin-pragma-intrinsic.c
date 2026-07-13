// RUN: %clang_cc1 -fms-extensions -fsyntax-only -verify -triple arm64-windows -isystem %S/Inputs %s -DUSE_PRAGMA_BEFORE
// RUN: %clang_cc1 -fms-extensions -fsyntax-only -verify -triple arm64-windows -isystem %S/Inputs %s -DUSE_PRAGMA_AFTER
// RUN: %clang_cc1 -fms-extensions -fsyntax-only -verify -triple arm64-windows -isystem %S/Inputs %s -DUSE_PRAGMA_AFTER_USE
// RUN: %clang_cc1 -fms-extensions -fsyntax-only -verify -triple arm64-windows -isystem %S/Inputs %s -DUSE_PRAGMA_SAME_FILE
// RUN: %clang_cc1 -fms-extensions -fsyntax-only -verify -triple arm64-windows -isystem %S/Inputs %s

#if defined(USE_PRAGMA_BEFORE) || defined(USE_PRAGMA_AFTER) || defined(USE_PRAGMA_SAME_FILE)
// expected-no-diagnostics
#else
// expected-error@+10 {{call to undeclared library function '_InterlockedOr64'}}
// expected-note@+9 {{include the header <intrin.h> or explicitly provide a declaration for '_InterlockedOr64'}}
#endif
#include <builtin-system-header.h>

#ifdef USE_PRAGMA_SAME_FILE
#pragma intrinsic(_InterlockedOr64)
#endif

void foo() {
  MACRO(0,0);
}

#ifdef USE_PRAGMA_AFTER_USE
#pragma intrinsic(_InterlockedOr64)
#endif
