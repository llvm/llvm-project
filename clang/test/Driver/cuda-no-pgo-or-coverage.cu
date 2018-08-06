// Check that profiling/coverage arguments doen't get passed down to device-side
// compilation.
//
// REQUIRES: clang-driver
//
// XRUN: %clang -### -target x86_64-linux-gnu -c --cuda-gpu-arch=sm_20 \
// XRUN:   -fprofile-generate %s 2>&1 | \
// XRUN: FileCheck --check-prefixes=CHECK,PROF %s
//
// RUN: %clang -### -target x86_64-linux-gnu -c --cuda-gpu-arch=sm_20 \
// RUN:   -fprofile-instr-generate %s 2>&1 | \
// RUN: FileCheck -allow-deprecated-dag-overlap --check-prefixes=CHECK,PROF %s
//
// RUN: %clang -### -target x86_64-linux-gnu -c --cuda-gpu-arch=sm_20 \
// RUN:   -coverage %s 2>&1 | \
// RUN: FileCheck -allow-deprecated-dag-overlap --check-prefixes=CHECK,GCOV %s
//
// RUN: %clang -### -target x86_64-linux-gnu -c --cuda-gpu-arch=sm_20 \
// RUN:   -ftest-coverage %s 2>&1 | \
// RUN: FileCheck -allow-deprecated-dag-overlap --check-prefixes=CHECK,GCOV %s
//
// RUN: %clang -### -target x86_64-linux-gnu -c --cuda-gpu-arch=sm_20   \
// RUN:   -fprofile-instr-generate -fcoverage-mapping %s 2>&1 | \
// RUN: FileCheck -allow-deprecated-dag-overlap --check-prefixes=CHECK,PROF,GCOV %s
//
//
// CHECK-NOT: error: unsupported option '-fprofile
// CHECK-NOT: error: invalid argument
// CHECK-DAG: "-fcuda-is-device"
// CHECK-NOT: "-f{{[^"]*coverage.*}}"
// CHECK-NOT: "-fprofile{{[^"]*}}"
// CHECK: "-triple" "x86_64--linux-gnu"
// PROF-DAG: "-fprofile{{.*}}"
// GCOV-DAG: "-f{{(coverage|emit-coverage).*}}"
