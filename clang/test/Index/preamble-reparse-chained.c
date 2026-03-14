// RUN: c-index-test -write-pch %t.h.pch -x c-header %S/Inputs/a.h
// RUN: env CINDEXTEST_EDITING=1 c-index-test -test-load-source-reparse 5 all -I%S/Inputs -include %t.h %s | FileCheck %s
#include "a.h"
#include "b.h"

A a;
B b;

// CHECK: a.h:3:13: TypedefDecl=A:3:13 (Definition) Extent=[3:1 - 3:14]
// CHECK: b.h:1:15: TypedefDecl=B:1:15 (Definition) Extent=[1:1 - 1:16]
