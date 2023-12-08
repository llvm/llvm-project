// RUN: rm -rf %t.dir
// RUN: rm -rf %t.cdb
// RUN: mkdir -p %t.dir
// RUN: cp %s %t.dir/preprocess_minimized_pragmas_basic.cpp
// RUN: cp %s %t.dir/preprocess_minimized_pragmas_ms.cpp
// RUN: mkdir %t.dir/Inputs
// RUN: cp %S/Inputs/preprocess_minimized_pragmas.h %t.dir/Inputs/preprocess_minimized_pragmas.h
// RUN: touch %t.dir/Inputs/a.h
// RUN: touch %t.dir/Inputs/b.h
// RUN: touch %t.dir/Inputs/c.h
// RUN: touch %t.dir/Inputs/c_alias.h
// RUN: sed -e "s|DIR|%/t.dir|g" %S/Inputs/preprocess_minimized_pragmas_cdb.json > %t.cdb
//
// RUN: clang-scan-deps -compilation-database %t.cdb -j 1 -mode preprocess-dependency-directives | \
// RUN:   FileCheck %s
// RUN: clang-scan-deps -compilation-database %t.cdb -j 1 -mode preprocess | \
// RUN:   FileCheck %s

#include "preprocess_minimized_pragmas.h"

// CHECK: preprocess_minimized_pragmas_basic.cpp
// CHECK-NEXT: Inputs{{/|\\}}preprocess_minimized_pragmas.h
// CHECK-NEXT: Inputs{{/|\\}}a.h
// CHECK-NEXT: Inputs{{/|\\}}b.h
// Expect include aliasing alias "c_alias.h" -> "c.h" to fail when Microsoft extensions are off.
// CHECK-NEXT: Inputs{{/|\\}}c_alias.h

// CHECK: preprocess_minimized_pragmas_ms.cpp
// CHECK-NEXT: Inputs{{/|\\}}preprocess_minimized_pragmas.h
// CHECK-NEXT: Inputs{{/|\\}}a.h
// CHECK-NEXT: Inputs{{/|\\}}b.h
// CHECK-NEXT: Inputs{{/|\\}}c.h
