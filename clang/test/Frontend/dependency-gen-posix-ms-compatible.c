// REQUIRES: !system-windows
// RUN: rm -rf %t.dir
// RUN: mkdir -p %t.dir/include/foo
// RUN: echo > %t.dir/include/foo/bar.h
// RUN: echo > %t.dir/include/foo\\bar.h

// RUN: %clang -MD -MF - %s -fsyntax-only -fms-compatibility -I %t.dir/include | FileCheck -check-prefix=CHECK-ONE %s
// CHECK-ONE: foo/bar.h
// CHECK-ONE-NOT: foo\bar.h
// CHECK-ONE-NOT: foo\\bar.h

// RUN: %clang -MD -MF - %s -fsyntax-only -fno-ms-compatibility -I %t.dir/include | FileCheck -check-prefix=CHECK-TWO %s
// CHECK-TWO: foo\bar.h
// CHECK-TWO-NOT: foo/bar.h
// CHECK-TWO-NOT: foo\\bar.h

#include "foo\bar.h"
