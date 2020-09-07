// RUN: %clangxx -target x86_64-linux-gnu -fmemory-profile %s -### 2>&1 | FileCheck %s
// RUN: %clangxx -target x86_64-linux-gnu -fmemory-profile -fno-memory-profile %s -### 2>&1 | FileCheck %s --check-prefix=OFF
// CHECK: "-cc1" {{.*}} "-fmemory-profile"
// CHECK: ld{{.*}}libclang_rt.heapprof{{.*}}libclang_rt.heapprof_cxx
// OFF-NOT: "-fmemory-profile"
// OFF-NOT: libclang_rt.heapprof
