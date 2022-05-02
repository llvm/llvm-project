// RUN: env CC_PRINT_PROC_STAT=1 \
// RUN:     CC_PRINT_PROC_STAT_FILE=%t.csv \
// RUN: %clang -no-canonical-prefixes -S -o %t.s %s
// RUN: FileCheck --check-prefix=CHECK-CSV %s < %t.csv
// CHECK-CSV: clang{{.*}},"{{.*}}.s",{{[0-9]+}},{{[0-9]+}},{{[0-9]+}}

// RUN: env CC_PRINT_PROC_STAT=1 \
// RUN: %clang -no-canonical-prefixes -c -fintegrated-as %s | FileCheck %s
// CHECK: clang{{.*}}: output={{.*}}.o, total={{[0-9.]+}} ms, user={{[0-9.]+}} ms, mem={{[0-9]+}} Kb
