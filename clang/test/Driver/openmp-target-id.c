// REQUIRES: clang-driver
// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target

//
// Legacy mode (-fopenmp-targets,-Xopenmp-target,-march) tests for TargetID
//
// RUN:   %clang -### -target x86_64-linux-gnu -fopenmp\
// RUN:   -fopenmp-targets=amdgcn-amd-amdhsa,amdgcn-amd-amdhsa \
// RUN:   -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx908:xnack+:sramecc+ \
// RUN:   -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx908:xnack+:sramecc- \
// RUN:   %s 2>&1 | FileCheck %s

// RUN:   %clang -### -target x86_64-linux-gnu -fopenmp\
// RUN:   -fopenmp-targets=amdgcn-amd-amdhsa,amdgcn-amd-amdhsa \
// RUN:   -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx908:xnack+:sramecc+ \
// RUN:   -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx908:xnack+:sramecc- \
// RUN:   -save-temps \
// RUN:   %s 2>&1 | FileCheck %s

// RUN:   %clang -### -target x86_64-linux-gnu -fopenmp\
// RUN:   -fopenmp-targets=amdgcn-amd-amdhsa,amdgcn-amd-amdhsa \
// RUN:   -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx908:xnack+:sramecc+ \
// RUN:   -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx908:xnack+:sramecc- \
// RUN:   -fgpu-rdc \
// RUN:   %s 2>&1 | FileCheck %s

//
// Offload-arch mode (--offload-arch) tests for TargetID
//
// RUN:   %clang -### -target x86_64-linux-gnu -fopenmp\
// RUN:   --offload-arch=gfx908:xnack+:sramecc+ \
// RUN:   --offload-arch=gfx908:xnack+:sramecc- \
// RUN:   %s 2>&1 | FileCheck %s

// RUN:   %clang -### -target x86_64-linux-gnu -fopenmp\
// RUN:   --offload-arch=gfx908:xnack+:sramecc+ \
// RUN:   --offload-arch=gfx908:xnack+:sramecc- \
// RUN:   -save-temps \
// RUN:   %s 2>&1 | FileCheck %s

// RUN:   %clang -### -target x86_64-linux-gnu -fopenmp\
// RUN:   --offload-arch=gfx908:xnack+:sramecc+ \
// RUN:   --offload-arch=gfx908:xnack+:sramecc- \
// RUN:   -fgpu-rdc \
// RUN:   %s 2>&1 | FileCheck %s

// CHECK: [[CLANG:"[^"]*clang[^"]*"]] "-cc1" "-triple" "amdgcn-amd-amdhsa"
// CHECK-SAME: "-target-cpu" "gfx908"
// CHECK-SAME: "-target-feature" "+sramecc"
// CHECK-SAME: "-target-feature" "+xnack"

// CHECK: [[OPT:"[^"]*opt[^"]*"]] {{.*}}  "-mcpu=gfx908"
// CHECK-SAME: "-mattr=+sramecc,+xnack"

// CHECK: [[LLC:"[^"]*llc[^"]*"]] {{.*}}  "-mcpu=gfx908"
// CHECK-SAME: "-mattr=+sramecc,+xnack

// CHECK: [[LLD:"[^"]*lld[^"]*"]] {{.*}} "-plugin-opt=mcpu=gfx908"
// CHECK-SAME: "-plugin-opt=-mattr=+sramecc,+xnack"

// CHECK: [[CLANG]] "-cc1" "-triple" "amdgcn-amd-amdhsa"
// CHECK-SAME: "-target-cpu" "gfx908"
// CHECK-SAME: "-target-feature" "-sramecc"
// CHECK-SAME: "-target-feature" "+xnack"

// CHECK: [[OPT:"[^"]*opt[^"]*"]] {{.*}}  "-mcpu=gfx908"
// CHECK-SAME: "-mattr=-sramecc,+xnack"

// CHECK: [[LLC:"[^"]*llc[^"]*"]] {{.*}}  "-mcpu=gfx908"
// CHECK-SAME: "-mattr=-sramecc,+xnack

// CHECK: [[LLD]] {{.*}} "-plugin-opt=mcpu=gfx908"
// CHECK-SAME: "-plugin-opt=-mattr=-sramecc,+xnack"

// CHECK: {{"[^"]*clang-offload-wrapper[^"]*"}}
// CHECK-SAME: "-target" "x86_64-unknown-linux-gnu" {{.*}} "--offload-arch=gfx908:sramecc+:xnack+" {{.*}} "--offload-arch=gfx908:sramecc-:xnack+"
