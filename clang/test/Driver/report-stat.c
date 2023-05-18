// RUN: %clang -no-canonical-prefixes -c -fproc-stat-report -fintegrated-as %s -o %t.o | FileCheck %s
// CHECK: clang{{.*}}: output={{.*}}.o, total={{[0-9.]+}} ms, user={{[0-9.]+}} ms, mem={{[0-9]+}} Kb

// RUN: %clang -no-canonical-prefixes -c -fintegrated-as -fproc-stat-report=%t %s -o %t.o
// RUN: cat %t | FileCheck --check-prefix=CSV %s
// CSV: clang{{.*}},"{{.*}}.o",{{[0-9]+}},{{[0-9]+}},{{[0-9]+}}
