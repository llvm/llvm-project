// RUN: %clang -funique-source-file-names -### %s 2> %t
// RUN: FileCheck < %t %s

// CHECK: "-cc1"
// CHECK: "-funique-source-file-names"
