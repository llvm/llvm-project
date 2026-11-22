// RUN: %clang_dxc -T cs_6_3 -### %s 2>&1 | FileCheck %s

// CHECK: "-load-bool-from-mem=nonzero"
