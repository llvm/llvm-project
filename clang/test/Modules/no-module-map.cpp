// RUN: %clang_cc1 -fmodules-ts -fmodule-name=ab -x c++-header %S/Inputs/no-module-map/a.h %S/Inputs/no-module-map/b.h -emit-header-module -o %t.pcm
// RUN: %clang_cc1 -fmodules-ts -fmodule-file=%t.pcm %s -I%S/Inputs/no-module-map -verify
// RUN: %clang_cc1 -fmodules-ts -fmodule-file=%t.pcm %s -I%S/Inputs/no-module-map -verify -DA
// RUN: %clang_cc1 -fmodules-ts -fmodule-file=%t.pcm %s -I%S/Inputs/no-module-map -verify -DB
// RUN: %clang_cc1 -fmodules-ts -fmodule-file=%t.pcm %s -I%S/Inputs/no-module-map -verify -DA -DB

// RUN: %clang_cc1 -E %t.pcm -o - | FileCheck %s
// RUN: %clang_cc1 -frewrite-imports -E %t.pcm -o - | FileCheck %s
// CHECK: # {{.*}}a.h
// CHECK: # {{.*}}b.h

#ifdef B
// expected-no-diagnostics
#endif

#ifdef A
#include "a.h"
#endif

#ifdef B
#include "b.h"
#endif

#if defined(A) || defined(B)
#ifndef A_H
#error A_H should be defined
#endif
#else
#ifdef A_H
#error A_H should not be defined
#endif
// expected-error@+3 {{must be imported from}}
// expected-note@* {{previous declaration}}
#endif
void use_a() { a(); }

#if defined(B)
#ifndef B_H
#error B_H should be defined
#endif
#else
#ifdef B_H
#error B_H should not be defined
#endif
// expected-error@+3 {{must be imported from}}
// expected-note@* {{previous declaration}}
#endif
void use_b() { b(); }
