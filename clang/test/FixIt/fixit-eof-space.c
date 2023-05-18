// RUN: not %clang_cc1 %s -fsyntax-only -fdiagnostics-parseable-fixits 2>&1 | FileCheck %s
// vim: set binary noeol:

// This file intentionally ends without a \n on the last line.  Make sure your
// editor doesn't add one. The trailing space is also intentional.

// CHECK: :9:8: warning: duplicate 'extern' declaration specifier
// CHECK: fix-it:"{{.*}}":{9:8-9:15}:""
extern extern 