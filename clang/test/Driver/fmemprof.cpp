// RUN: %clangxx -target x86_64-linux-gnu -fmemprof %s -### 2>&1 | FileCheck %s
// RUN: %clangxx -target x86_64-linux-gnu -fmemprof -fno-memprof %s -### 2>&1 | FileCheck %s --check-prefix=OFF
// CHECK: "-cc1" {{.*}} "-fmemprof"
// CHECK: ld{{.*}}libclang_rt.heapprof{{.*}}libclang_rt.heapprof_cxx
// OFF-NOT: "-fmemprof"
// OFF-NOT: libclang_rt.heapprof
