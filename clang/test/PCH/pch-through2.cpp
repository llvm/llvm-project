// Create
// RUN: %clang_cc1 -I %S -emit-pch \
// RUN:   -pch-through-header=Inputs/pch-through2.h -o %t.1 %s

// Use
// RUN: %clang_cc1 -I %S -include-pch %t.1 \
// RUN:   -pch-through-header=Inputs/pch-through2.h %s

// No #include of through header during pch use
// RUN: not %clang_cc1 -I %S -include-pch %t.1 \
// RUN:   -pch-through-header=Inputs/pch-through2.h \
// RUN:   %S/Inputs/pch-through-use1.cpp 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-TEST2A %s
// CHECK-TEST2A: fatal error:{{.*}} #include of
// CHECK-TEST2A-SAME: 'Inputs/pch-through2.h' not seen while attempting to
// CHECK-TEST2A-SAME: use precompiled header

// check that pch only contains code before the through header.
// RUN: %clang_cc1 -I %S -emit-pch \
// RUN:   -pch-through-header=Inputs/pch-through1.h -o %t.2 %s
// RUN: not %clang_cc1 -I %S -include-pch %t.2 \
// RUN:   -pch-through-header=Inputs/pch-through1.h \
// RUN:   %S/Inputs/pch-through-use1.cpp 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-TEST3 %s
// CHECK-TEST3: error: use of undeclared identifier 'through2'

#include "Inputs/pch-through1.h"
#include "Inputs/pch-through2.h"
