// REQUIRES: case-insensitive-filesystem

// RUN: mkdir -p %t/asystempath
// RUN: cp %S/Inputs/case-insensitive-include.h %t/asystempath/
// RUN: cd %t
// RUN: %clang_cc1 -fsyntax-only %s -include %s -isystem %t/asystempath -verify -Wnonportable-system-include-path
// RUN: %clang_cc1 -fsyntax-only -fdiagnostics-parseable-fixits %s -include %s -isystem %t/asystempath -Wnonportable-system-include-path 2>&1 | FileCheck %s

#include "CASE-INSENSITIVE-INCLUDE.H" // expected-warning {{non-portable path}}
// CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:10-[[@LINE-1]]:38}:"\"case-insensitive-include.h\""
