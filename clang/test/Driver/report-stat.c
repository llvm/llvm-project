// RUN: %clang -c -fproc-stat-report %s | FileCheck %s
// CHECK: clang{{.*}}: output={{.*}}.o, total={{[0-9.]+}} ms, user={{[0-9.]+}} ms, mem={{[0-9]+}} Kb

// RUN: %clang -c -fproc-stat-report=%t %s
// RUN: cat %t | FileCheck --check-prefix=CSV %s
// CSV: clang{{.*}},"{{.*}}.o",{{[0-9]+}},{{[0-9]+}},{{[0-9]+}}
