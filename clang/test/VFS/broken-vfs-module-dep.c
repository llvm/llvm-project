// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: not %clang_cc1 -module-dependency-dir %t -ivfsoverlay %S/Inputs/invalid-yaml.yaml %s 2>&1 | FileCheck %s

// CHECK: error: Unexpected token
// CHECK: error: Unexpected token
// CHECK: 1 error generated
