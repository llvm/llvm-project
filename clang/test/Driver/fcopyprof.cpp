// Test driver flags for CopyProf instrumentation and runtime linking.

// Basic C and C++ invocations:
// RUN: %clang --target=x86_64-linux-gnu -fcopyprof %s -### 2>&1 | FileCheck %s
// RUN: %clangxx --target=x86_64-linux-gnu -fcopyprof %s -### 2>&1 | FileCheck %s
// CHECK: "-cc1" {{.*}} "-fcopyprof"
// CHECK: ld{{.*}}libclang_rt.copyprof

// Re-enabling after -fno-copyprof:
// RUN: %clangxx --target=x86_64-linux-gnu -fcopyprof -fno-copyprof -fcopyprof %s -### 2>&1 | FileCheck %s

// Custom static size threshold (and last-flag-wins behavior):
// RUN: %clangxx --target=x86_64-linux-gnu -fcopyprof -fcopyprof-static-size-threshold=32 %s -### 2>&1 | FileCheck %s --check-prefix=THRESHOLD32
// THRESHOLD32: "-cc1" {{.*}} "-fcopyprof" "-fcopyprof-static-size-threshold=32"
// THRESHOLD32: ld{{.*}}libclang_rt.copyprof

// RUN: %clangxx --target=x86_64-linux-gnu -fcopyprof -fcopyprof-static-size-threshold=32 -fcopyprof-static-size-threshold=64 %s -### 2>&1 | FileCheck %s --check-prefix=THRESHOLD64
// THRESHOLD64: "-cc1" {{.*}} "-fcopyprof" "-fcopyprof-static-size-threshold=64"
// THRESHOLD64-NOT: "-fcopyprof-static-size-threshold=32"

// Disabling CopyProf via -fno-copyprof (ensuring threshold is also suppressed):
// RUN: %clangxx --target=x86_64-linux-gnu -fcopyprof -fno-copyprof %s -### 2>&1 | FileCheck %s --check-prefix=OFF
// RUN: %clangxx --target=x86_64-linux-gnu -fcopyprof -fcopyprof-static-size-threshold=32 -fno-copyprof %s -### 2>&1 | FileCheck %s --check-prefix=OFF
// RUN: %clangxx --target=x86_64-linux-gnu -fcopyprof-static-size-threshold=32 %s -### 2>&1 | FileCheck %s --check-prefix=OFF
// OFF-NOT: "-fcopyprof"
// OFF-NOT: "-fcopyprof-static-size-threshold"
// OFF-NOT: libclang_rt.copyprof

// Shared libraries (-shared) and relocatable links (-r) compile with -fcopyprof
// but must not link the static CopyProf runtime archive:
// RUN: %clangxx --target=x86_64-linux-gnu -fcopyprof -shared %s -### 2>&1 | FileCheck %s --check-prefix=NO-STATIC-RT
// RUN: %clangxx --target=x86_64-linux-gnu -fcopyprof -r %s -### 2>&1 | FileCheck %s --check-prefix=NO-STATIC-RT
// NO-STATIC-RT: "-cc1" {{.*}} "-fcopyprof"
// NO-STATIC-RT-NOT: libclang_rt.copyprof
