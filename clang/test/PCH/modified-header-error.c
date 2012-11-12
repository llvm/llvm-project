// RUN: mkdir -p %t.dir
// RUN: echo '#include "header2.h"' > %t.dir/header1.h
// RUN: echo > %t.dir/header2.h
// RUN: cp %s %t.dir/t.c
// RUN: %clang_cc1 -x c-header %t.dir/header1.h -emit-pch -o %t.pch
// RUN: echo >> %t.dir/header2.h
// RUN: %clang_cc1 %t.dir/t.c -include-pch %t.pch -fsyntax-only 2>&1 | FileCheck %s

#include "header2.h"

// CHECK: fatal error: file {{.*}} has been modified since the precompiled header was built
// REQUIRES: shell
