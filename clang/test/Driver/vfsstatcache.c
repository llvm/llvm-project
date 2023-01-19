// RUN: %clang -ivfsstatcache foo.h -### %s 2>&1 | FileCheck %s
// CHECK: "-ivfsstatcache" "foo.h"

// RUN: not %clang -ivfsstatcache foo.h %s 2>&1 | FileCheck -check-prefix=CHECK-MISSING %s
// CHECK-MISSING: stat cache file 'foo.h' not found
