// RUN: %clang -### -fsycl -c %s 2>&1 | FileCheck %s --check-prefix=ENABLED
// RUN: %clang -### -fsycl %s 2>&1 | FileCheck %s --check-prefix=ENABLED
// RUN: %clang -### -fno-sycl -fsycl %s 2>&1 | FileCheck %s --check-prefix=ENABLED
// RUN: %clangxx -### -fsycl %s 2>&1 | FileCheck %s --check-prefix=ENABLED
// RUN: %clangxx -### -fno-sycl %s 2>&1 | FileCheck %s --check-prefix=DISABLED
// RUN: %clangxx -### -fsycl -fno-sycl %s 2>&1 | FileCheck %s --check-prefix=DISABLED
// RUN: %clangxx -### %s 2>&1 | FileCheck %s --check-prefix=DISABLED

// ENABLED: "-cc1"{{.*}} "-fsycl-is-device"
// DISABLED-NOT: "-fsycl-is-device"
