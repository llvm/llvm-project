// RUN: clang-format -assume-filename=foo.cl -dump-config | FileCheck %s

// RUN: clang-format -dump-config - < %s | FileCheck %s

// CHECK: Language: C

void foo() {}
