// RUN: %clang -ivfsoverlay foo.h -### %s 2>&1 | FileCheck %s
// CHECK: "-ivfsoverlay" "foo.h"

// RUN: %clang --vfsoverlay foo.h -### %s 2>&1 | FileCheck %s --check-prefix=CHECK2
// CHECK2: "--vfsoverlay" "foo.h"

// RUN: not %clang -ivfsoverlay foo.h %s 2>&1 | FileCheck -check-prefix=CHECK-MISSING %s
// CHECK-MISSING: virtual filesystem overlay file 'foo.h' not found

// The driver applies overlays.
// RUN: rm -rf %t && mkdir -p %t/real
// RUN: echo "int a;" > %t/real/a.c
// RUN: echo "{'version': 0, 'roots': [{'type': 'directory-remap', 'name': '%/t/virtual', 'external-contents': '%/t/real'}]}" > %t/vfs.yaml
// RUN: not %clang -fsyntax-only %t/virtual/a.c 2>&1 | FileCheck %s --check-prefix=CHECK-NO-OVERLAY
// CHECK-NO-OVERLAY: error: no such file or directory: '{{.*}}a.c'
// RUN: %clang -fsyntax-only -ivfsoverlay %t/vfs.yaml %t/virtual/a.c
// RUN: %clang -fsyntax-only -vfsoverlay %t/vfs.yaml %t/virtual/a.c

// The driver leaves invalid overlays for -cc1 to diagnose.
// RUN: echo "invalid" > %t/invalid.yaml
// RUN: not %clang -fsyntax-only -ivfsoverlay %t/invalid.yaml %t/real/a.c 2>&1 | FileCheck -check-prefix=CHECK-INVALID %s
// RUN: not %clang -fsyntax-only -vfsoverlay %t/invalid.yaml %t/real/a.c 2>&1 | FileCheck -check-prefix=CHECK-INVALID %s
// CHECK-INVALID: invalid virtual filesystem overlay file '{{.*}}invalid.yaml'
