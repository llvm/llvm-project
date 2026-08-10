// RUN: %clang -ivfsoverlay foo.h -### %s 2>&1 | FileCheck %s
// CHECK: "-ivfsoverlay" "foo.h"

// RUN: %clang --vfsoverlay foo.h -### %s 2>&1 | FileCheck %s --check-prefix=CHECK2
// CHECK2: "--vfsoverlay" "foo.h"

// RUN: not %clang -ivfsoverlay foo.h %s 2>&1 | FileCheck -check-prefix=CHECK-MISSING %s
// CHECK-MISSING: virtual filesystem overlay file 'foo.h' not found
