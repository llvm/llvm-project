// RUN: inter-compile-api-test %inter_pipelines | FileCheck %s

// CHECK: zebin: {{[1-9][0-9]*}} bytes
// CHECK-NEXT: ged: {{[1-9][0-9]*}} bytes
// CHECK-NEXT: assembly: present
// CHECK-NEXT: validation: passed
// CHECK: diagnostic: {{.*}}unsupported LLVM target triple 'x86_64-unknown-linux-gnu'
