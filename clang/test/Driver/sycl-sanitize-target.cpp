// UNSUPPORTED: system-windows

// Check whether ASan runtime libraries are linked or not when
// fsanitize-target=host|device|both. The ASan runtime libraries
// are NOT needed for device compilation in offloading scenario.

// RUN: %clangxx -fsycl -fsanitize=address %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=SYCL-ASAN-DEFAULT %s
// SYCL-ASAN-DEFAULT:  "{{.*}}/ld"
// SYCL-ASAN-DEFAULT-SAME: "{{.*}}/libclang_rt.asan{{.*}}.a"

// RUN: %clangxx -fsycl -fsanitize=address -fsanitize-target=host %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=SYCL-ASAN-HOST %s
// SYCL-ASAN-HOST:  "{{.*}}/ld"
// SYCL-ASAN-HOST-SAME: "{{.*}}/libclang_rt.asan{{.*}}.a"

// RUN: %clangxx -fsycl -fsanitize=address -fsanitize-target=both %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=SYCL-ASAN-BOTH %s
// SYCL-ASAN-BOTH:  "{{.*}}/ld"
// SYCL-ASAN-BOTH-SAME: "{{.*}}/libclang_rt.asan{{.*}}.a"

// RUN: %clangxx -fsycl -fsanitize=address -fsanitize-target=device %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=SYCL-ASAN-DEVICE %s
// SYCL-ASAN-DEVICE:  "{{.*}}/ld"
// SYCL-ASAN-DEVICE-NOT: "{{.*}}/libclang_rt.asan{{.*}}.a"

// RUN: not %clangxx -fsycl -fsanitize=address -fsanitize-target=YYY %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=SYCL-ASAN-INVALID %s
// SYCL-ASAN-INVALID: error: unsupported argument 'YYY'
