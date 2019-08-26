// REQUIRES: x86-registered-target
// RUN: rm -rf %t.cdb
// RUN: %clang -target x86_64-apple-macos10.15 -c %s -o -  -gen-cdb-fragment-path %t.cdb
// RUN: ls %t.cdb | FileCheck --check-prefix=CHECK-LS %s
// CHECK-LS: gen-cdb-fragment.c.{{.*}}.json

// RUN: cat %t.cdb/*.json | FileCheck --check-prefix=CHECK %s
// CHECK: { "directory": "{{.*}}", "file": "{{.*}}gen-cdb-fragment.c", "output": "-", "arguments": [{{.*}}, "--target=x86_64-apple-macos10.15"{{.*}}]},
// RUN: cat %t.cdb/*.json | FileCheck --check-prefix=CHECK-FLAG %s
// CHECK-FLAG-NOT: -gen-cdb-fragment-path

// RUN: rm -rf %t.cdb
// RUN: mkdir %t.cdb
// RUN: ls %t.cdb | not FileCheck --check-prefix=CHECK-LS %s
// RUN: %clang -target x86_64-apple-macos10.15 -S %s -o -  -gen-cdb-fragment-path %t.cdb
// RUN: ls %t.cdb | FileCheck --check-prefix=CHECK-LS %s

// Working directory arg is respected.
// RUN: rm -rf %t.cdb
// RUN: mkdir %t.cdb
// RUN: %clang -target x86_64-apple-macos10.15 -working-directory %t.cdb -c %s -o -  -gen-cdb-fragment-path "."
// RUN: ls %t.cdb | FileCheck --check-prefix=CHECK-LS %s
// RUN: cat %t.cdb/*.json | FileCheck --check-prefix=CHECK-CWD %s
// CHECK-CWD: "directory": "{{.*}}.cdb"

// -### does not emit the CDB fragment
// RUN: rm -rf %t.cdb
// RUN: mkdir %t.cdb
// RUN: %clang -target x86_64-apple-macos10.15 -S %s -o -  -gen-cdb-fragment-path %t.cdb -###
// RUN: ls %t.cdb | not FileCheck --check-prefix=CHECK-LS %s

// -MJ is preferred over -gen-cdb-fragment-path
// RUN: rm -rf %t.cdb
// RUN: mkdir %t.cdb
// RUN: %clang -target x86_64-apple-macos10.15 -S %s -o -  -gen-cdb-fragment-path %t.cdb -MJ %t.out
// RUN: ls %t.cdb | not FileCheck --check-prefix=CHECK-LS %s
// RUN: FileCheck %s < %t.out
