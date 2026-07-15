// REQUIRES: amdgpu-registered-target, x86-registered-target

// Check that target ID features (xnack, sramecc) requested via -march or
// --offload-arch are propagated to the device -cc1 compilation and recorded in
// the embedded offload binary metadata.

//
// Legacy mode (-fopenmp-targets, -Xopenmp-target, -march) for TargetID
//
// RUN:   %clang -### -target x86_64-linux-gnu -fopenmp -nogpulib \
// RUN:   -fopenmp-targets=amdgcn-amd-amdhsa \
// RUN:   -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx908:xnack+:sramecc- \
// RUN:   %s 2>&1 | FileCheck -check-prefix=LEGACY %s

// LEGACY: [[CLANG:"[^"]*clang[^"]*"]] "-cc1" "-triple" "amdgcn-amd-amdhsa"
// LEGACY-SAME: "-target-cpu" "gfx908"
// LEGACY-SAME: "-target-feature" "+xnack"
// LEGACY-SAME: "-target-feature" "-sramecc"

//
// Offload-arch mode (--offload-arch) for TargetID
//
// RUN:   %clang -### -target x86_64-linux-gnu -fopenmp -nogpulib \
// RUN:   --offload-arch=gfx908:xnack+:sramecc+ \
// RUN:   --offload-arch=gfx908:xnack+:sramecc- \
// RUN:   %s 2>&1 | FileCheck -check-prefix=OFFLOAD %s

// OFFLOAD: [[CLANG:"[^"]*clang[^"]*"]] "-cc1" "-triple" "amdgcn-amd-amdhsa"
// OFFLOAD-SAME: "-target-cpu" "gfx908"
// OFFLOAD-SAME: "-target-feature" "+xnack"
// OFFLOAD-SAME: "-target-feature" "+sramecc"

// OFFLOAD: [[CLANG]] "-cc1" "-triple" "amdgcn-amd-amdhsa"
// OFFLOAD-SAME: "-target-cpu" "gfx908"
// OFFLOAD-SAME: "-target-feature" "+xnack"
// OFFLOAD-SAME: "-target-feature" "-sramecc"

// OFFLOAD: "--image=file={{.*}},triple=amdgcn-amd-amdhsa,arch=gfx908:sramecc+:xnack+,kind=openmp,feature=+xnack,feature=+sramecc"
// OFFLOAD-SAME: "--image=file={{.*}},triple=amdgcn-amd-amdhsa,arch=gfx908:sramecc-:xnack+,kind=openmp,feature=+xnack,feature=-sramecc"
