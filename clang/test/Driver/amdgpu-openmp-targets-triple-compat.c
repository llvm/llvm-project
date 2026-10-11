/// Check amdgpu triples with a subarch respect -Xopenmp-target
// RUN:   %clang -### -no-canonical-prefixes -nogpulib -fopenmp=libomp -fopenmp-targets=amdgpu9-amd-amdhsa -Xopenmp-target=amdgpu9-amd-amdhsa -ffinite-math-only %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHECK,GFX9 %s

// RUN:   %clang -### -no-canonical-prefixes -nogpulib -fopenmp=libomp -fopenmp-targets=amdgpu9-amd-amdhsa -Xopenmp-target=amdgpu-amd-amdhsa -ffinite-math-only %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHECK,GFX9 %s

// RUN:   %clang -### -no-canonical-prefixes -nogpulib -fopenmp=libomp -fopenmp-targets=amdgpu9.00-amd-amdhsa -Xopenmp-target=amdgpu9-amd-amdhsa -ffinite-math-only %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHECK,GFX900 %s


// Check the same, with -Xarch instead of -Xopenmp-target
// RUN:   %clang -### -no-canonical-prefixes -nogpulib -fopenmp=libomp -fopenmp-targets=amdgpu9-amd-amdhsa -Xarch_amdgpu9 -ffinite-math-only %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHECK,GFX9 %s

// RUN:   %clang -### -no-canonical-prefixes -nogpulib -fopenmp=libomp -fopenmp-targets=amdgpu9-amd-amdhsa -Xarch_amdgpu -ffinite-math-only %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHECK,GFX9 %s

// RUN:   %clang -### -no-canonical-prefixes -nogpulib -fopenmp=libomp -fopenmp-targets=amdgpu9.00-amd-amdhsa -Xarch_amdgpu9 -ffinite-math-only %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHECK,GFX900 %s


// GFX9: clang{{.*}} "-cc1" "-triple" "amdgpu9-amd-amdhsa"
// GFX900: clang{{.*}} "-cc1" "-triple" "amdgpu9.00-amd-amdhsa"

// CHECK-SAME: "-ffinite-math-only"
// CHECK-SAME: "-fopenmp-is-target-device"

/// ###########################################################################

/// Check amdgpu triples with a subarch do not forward to an incompatible subarch
// RUN:   %clang -### -no-canonical-prefixes -nogpulib -fopenmp=libomp -fopenmp-targets=amdgpu9.00-amd-amdhsa -Xopenmp-target=amdgpu9.08-amd-amdhsa -ffinite-math-only %s 2>&1 \
// RUN:   | FileCheck -check-prefix=MISMATCH-SUBARCH %s

// RUN:   %clang -### -no-canonical-prefixes -nogpulib -fopenmp=libomp -fopenmp-targets=amdgpu10.1-amd-amdhsa -Xopenmp-target=amdgpu9.08-amd-amdhsa -ffinite-math-only %s 2>&1 \
// RUN:   | FileCheck -check-prefix=MISMATCH-SUBARCH %s

// MISMATCH-SUBARCH-NOT: "-ffinite-math-only"

