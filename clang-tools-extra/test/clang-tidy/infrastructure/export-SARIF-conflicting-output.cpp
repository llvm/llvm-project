// RUN: not clang-tidy %s -checks='-*,google-explicit-constructor' -sarif-export=- -export-fixes=- -- 2>&1 | FileCheck %s --check-prefix=CHECK-STDOUT
// RUN: not clang-tidy %s -checks='-*,google-explicit-constructor' -sarif-export=%t.out -export-fixes=%t.out -- 2>&1 | FileCheck %s --check-prefix=CHECK-FILE

// CHECK-STDOUT: Error: --sarif-export and --export-fixes cannot write to the same output '-'.
// CHECK-STDOUT-NOT: warning:
// CHECK-FILE: Error: --sarif-export and --export-fixes cannot write to the same output '{{.*}}.out'.

struct A { A(int) {} };

