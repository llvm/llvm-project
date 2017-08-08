// RUN: %clang -### -frewrite-map-file %t.map -c %s -o /dev/null 2>&1 | FileCheck %s
// CHECK: error: no such file or directory:
