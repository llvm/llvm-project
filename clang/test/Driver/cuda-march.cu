// Checks that cuda compilation does the right thing when passed -march.
// (Specifically, we want to pass it to host compilation, but not to device
// compilation or ptxas!)

// RUN: %clang -### --target=x86_64-linux-gnu -c \
// RUN: -nogpulib -nogpuinc -march=haswell %s 2>&1 | FileCheck %s --check-prefix=CHECK-DEFAULT
// RUN: %clang -### --target=x86_64-linux-gnu -c \
// RUN: -nogpulib -nogpuinc -march=haswell --cuda-gpu-arch=sm_52 %s 2>&1 | FileCheck %s --check-prefix=CHECK-SM52

// CHECK-DEFAULT: "-cc1"{{.*}} "-triple" "nvptx
// CHECK-DEFAULT-SAME: "-target-cpu" "sm_75"

// CHECK-DEFAULT: ptxas
// CHECK-DEFAULT-SAME: "--gpu-name" "sm_75"

// CHECK-DEFAULT: "-cc1"{{.*}} "-target-cpu" "haswell"

// CHECK-SM52: "-cc1"{{.*}} "-triple" "nvptx
// CHECK-SM52-SAME: "-target-cpu" "sm_52"

// CHECK-SM52: ptxas
// CHECK-SM52-SAME: "--gpu-name" "sm_52"

// CHECK-SM52: "-cc1"{{.*}} "-target-cpu" "haswell"
