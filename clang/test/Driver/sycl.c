// RUN: %clang -### -fsycl -c %s 2>&1 | FileCheck %s --check-prefix=ENABLED
// RUN: %clang -### -fsycl %s 2>&1 | FileCheck %s --check-prefix=ENABLED
// RUN: %clang -### -fsycl -sycl-std=1.2.1 %s 2>&1 | FileCheck %s --check-prefix=ENABLED
// RUN: %clang -### -fsycl -sycl-std=121 %s 2>&1 | FileCheck %s --check-prefix=ENABLED
// RUN: %clang -### -fsycl -sycl-std=2017 %s 2>&1 | FileCheck %s --check-prefix=ENABLED
// RUN: %clang -### -fsycl -sycl-std=2020 %s 2>&1 | FileCheck %s --check-prefix=ENABLED
// RUN: %clang -### -fsycl -sycl-std=sycl-1.2.1 %s 2>&1 | FileCheck %s --check-prefix=ENABLED
// RUN: %clang -### -fno-sycl -fsycl %s 2>&1 | FileCheck %s --check-prefix=ENABLED
// RUN: %clang -### -sycl-std=2017 %s 2>&1 | FileCheck %s --check-prefix=DISABLED
// RUN: %clangxx -### -fsycl %s 2>&1 | FileCheck %s --check-prefix=ENABLED
// RUN: %clangxx -### -fno-sycl %s 2>&1 | FileCheck %s --check-prefix=DISABLED
// RUN: %clangxx -### -fsycl -fno-sycl %s 2>&1 | FileCheck %s --check-prefix=DISABLED
// RUN: %clangxx -### %s 2>&1 | FileCheck %s --check-prefix=DISABLED
// RUN: %clang_cl -### -fsycl -sycl-std=2017 -- %s 2>&1 | FileCheck %s --check-prefix=ENABLED
// RUN: %clang_cl -### -fsycl -- %s 2>&1 | FileCheck %s --check-prefix=ENABLED
// RUN: %clang_cl -### -- %s 2>&1 | FileCheck %s --check-prefix=DISABLED

// ENABLED: "-cc1"{{.*}} "-fsycl-is-device"
// ENABLED-SAME: "-sycl-std={{[-.sycl0-9]+}}"
// DISABLED-NOT: "-fsycl-is-device"
// DISABLED-NOT: "-sycl-std="

// RUN: %clang -### -fsycl  %s 2>&1 | FileCheck %s --check-prefix=DEFAULT
// RUN: %clangxx -### -fsycl %s 2>&1 | FileCheck %s --check-prefix=DEFAULT
// RUN: %clang_cl -### -fsycl -- %s 2>&1 | FileCheck %s --check-prefix=DEFAULT

// DEFAULT: "-sycl-std=2020"
