// REQUIRES: amdgpu-registered-target

// Default options
// RUN: %clang -### -x cl -target amdgcn-amd-amdhsa -c -emit-llvm --no-offloadlib %s 2>&1 | FileCheck -check-prefix=ASYNC-TABLES %s
// RUN: %clang -### -x hip --offload-arch=gfx90a --offload-device-only --no-offloadlib -c -emit-llvm %s 2>&1 | FileCheck -check-prefix=ASYNC-TABLES  %s
// RUN: %clang -### -fopenmp --offload-arch=gfx90a --offload-device-only --no-offloadlib -c -emit-llvm %s 2>&1 | FileCheck -check-prefix=ASYNC-TABLES  %s

// Explicitly enable sync-tables (somewhat surprisingly this is still preempted by the default-on async tables)
// RUN: %clang -### -x cl -target amdgcn-amd-amdhsa -c -emit-llvm --no-offloadlib -funwind-tables %s 2>&1 | FileCheck -check-prefix=ASYNC-TABLES %s
// RUN: %clang -### -x hip --offload-arch=gfx90a --offload-device-only --no-offloadlib -c -emit-llvm -funwind-tables %s 2>&1 | FileCheck -check-prefix=ASYNC-TABLES  %s
// RUN: %clang -### -fopenmp --offload-arch=gfx90a --offload-device-only --no-offloadlib -c -emit-llvm -funwind-tables %s 2>&1 | FileCheck -check-prefix=ASYNC-TABLES  %s

// Explicitly enable sync-tables and surpress default async-tables
// RUN: %clang -### -x cl -target amdgcn-amd-amdhsa -c -emit-llvm --no-offloadlib -funwind-tables -fno-asynchronous-unwind-tables %s 2>&1 | FileCheck -check-prefix=SYNC-TABLES %s
// RUN: %clang -### -x hip --offload-arch=gfx90a --offload-device-only --no-offloadlib -c -emit-llvm -funwind-tables -fno-asynchronous-unwind-tables %s 2>&1 | FileCheck -check-prefix=SYNC-TABLES  %s
// RUN: %clang -### -fopenmp --offload-arch=gfx90a --offload-device-only --no-offloadlib -c -emit-llvm -funwind-tables -fno-asynchronous-unwind-tables %s 2>&1 | FileCheck -check-prefix=SYNC-TABLES  %s

// Suppress the default async-tables
// RUN: %clang -### -x cl -target amdgcn-amd-amdhsa -c -emit-llvm --no-offloadlib -fno-asynchronous-unwind-tables %s 2>&1 | FileCheck -check-prefix=NO-TABLES %s
// RUN: %clang -### -x hip --offload-arch=gfx90a --offload-device-only --no-offloadlib -c -emit-llvm -fno-asynchronous-unwind-tables %s 2>&1 | FileCheck -check-prefix=NO-TABLES  %s
// RUN: %clang -### -fopenmp --offload-arch=gfx90a --offload-device-only --no-offloadlib -c -emit-llvm -fno-asynchronous-unwind-tables %s 2>&1 | FileCheck -check-prefix=NO-TABLES  %s


// ASYNC-TABLES: "-triple" "amdgcn-amd-amdhsa" {{.*}} "-funwind-tables=2"
// SYNC-TABLES: "-triple" "amdgcn-amd-amdhsa" {{.*}} "-funwind-tables=1"
// NO-TABLES-NOT: "-triple" "amdgcn-amd-amdhsa" {{.*}} "-funwind-tables={{.*}}"
