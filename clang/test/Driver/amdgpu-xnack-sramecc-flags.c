// Test for -mxnack/-mno-xnack and -msramecc/-mno-sramecc flags, which should be
// forwarded to cc1.

// RUN: %clang -### --target=amdgcn-amd-amdhsa -mcpu=gfx90a -mxnack %s 2>&1 | \
// RUN:   FileCheck -check-prefix=XNACK-ON %s
// XNACK-ON: "-mxnack"

// RUN: %clang -### --target=amdgcn-amd-amdhsa -mcpu=gfx90a -mno-xnack %s 2>&1 | \
// RUN:   FileCheck -check-prefix=XNACK-OFF %s
// XNACK-OFF: "-mno-xnack"

// RUN: %clang -### --target=amdgcn-amd-amdhsa -mcpu=gfx90a -msramecc %s 2>&1 | \
// RUN:   FileCheck -check-prefix=SRAMECC-ON %s
// SRAMECC-ON: "-msramecc"

// RUN: %clang -### --target=amdgcn-amd-amdhsa -mcpu=gfx90a -mno-sramecc %s 2>&1 | \
// RUN:   FileCheck -check-prefix=SRAMECC-OFF %s
// SRAMECC-OFF: "-mno-sramecc"

// Test that target ID takes precedence over explicit flags
// RUN: %clang -### --target=amdgcn-amd-amdhsa -mcpu=gfx90a:xnack+ -mno-xnack %s 2>&1 | \
// RUN:   FileCheck -check-prefix=TARGETID-OVERRIDES-XNACK %s
// TARGETID-OVERRIDES-XNACK: "-mxnack"

// RUN: %clang -### --target=amdgcn-amd-amdhsa -mcpu=gfx90a:xnack- -mxnack %s 2>&1 | \
// RUN:   FileCheck -check-prefix=TARGETID-OVERRIDES-XNACK-OFF %s
// TARGETID-OVERRIDES-XNACK-OFF: "-mno-xnack"

// RUN: %clang -### --target=amdgcn-amd-amdhsa -mcpu=gfx90a:sramecc+ -mno-sramecc %s 2>&1 | \
// RUN:   FileCheck -check-prefix=TARGETID-OVERRIDES-SRAMECC %s
// TARGETID-OVERRIDES-SRAMECC: "-msramecc"

// RUN: %clang -### --target=amdgcn-amd-amdhsa -mcpu=gfx90a:sramecc- -msramecc %s 2>&1 | \
// RUN:   FileCheck -check-prefix=TARGETID-OVERRIDES-SRAMECC-OFF %s
// TARGETID-OVERRIDES-SRAMECC-OFF: "-mno-sramecc"

// Test combining both flags
// RUN: %clang -### --target=amdgcn-amd-amdhsa -mcpu=gfx90a -mxnack -msramecc %s 2>&1 | \
// RUN:   FileCheck -check-prefixes=BOTH-ON %s
// BOTH-ON: "-mxnack"
// BOTH-ON-SAME: "-msramecc"

// RUN: %clang -### --target=amdgcn-amd-amdhsa -mcpu=gfx90a -mno-xnack -mno-sramecc %s 2>&1 | \
// RUN:   FileCheck -check-prefixes=BOTH-OFF %s
// BOTH-OFF: "-mno-xnack"
// BOTH-OFF-SAME: "-mno-sramecc"

// Test that target ID without explicit features doesn't synthesize flags
// RUN: %clang -### --target=amdgcn-amd-amdhsa -mcpu=gfx90a %s 2>&1 | \
// RUN:   FileCheck -check-prefix=NO-FEATURES %s
// NO-FEATURES-NOT: "-mxnack"
// NO-FEATURES-NOT: "-mno-xnack"
// NO-FEATURES-NOT: "-msramecc"
// NO-FEATURES-NOT: "-mno-sramecc"

// Test target ID features are synthesized as flags
// RUN: %clang -### --target=amdgcn-amd-amdhsa -mcpu=gfx90a:xnack+ %s 2>&1 | \
// RUN:   FileCheck -check-prefix=TARGETID-XNACK %s
// TARGETID-XNACK: "-mxnack"

// RUN: %clang -### --target=amdgcn-amd-amdhsa -mcpu=gfx90a:sramecc+ %s 2>&1 | \
// RUN:   FileCheck -check-prefix=TARGETID-SRAMECC %s
// TARGETID-SRAMECC: "-msramecc"

// RUN: %clang -### --target=amdgcn-amd-amdhsa -mcpu=gfx90a:xnack+:sramecc+ %s 2>&1 | \
// RUN:   FileCheck -check-prefixes=TARGETID-BOTH %s
// TARGETID-BOTH: "-mxnack"
// TARGETID-BOTH-SAME: "-msramecc"

//
// Offload tests
//

// Test offload with target ID features synthesized from --offload-arch
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -fopenmp --offload-arch=gfx90a:xnack+:sramecc- \
// RUN:   -nogpulib %s 2>&1 | FileCheck -check-prefix=OMP-TARGETID %s
// OMP-TARGETID: "-cc1" "-triple" "amdgpu9.0a-amd-amdhsa" {{.*}} "-target-cpu" "gfx90a"
// OMP-TARGETID-SAME: "-mxnack"
// OMP-TARGETID-SAME: "-mno-sramecc"

// Test offload using -fopenmp-targets with target ID
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa \
// RUN:   -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx908:xnack-:sramecc+ \
// RUN:   -nogpulib %s 2>&1 | FileCheck -check-prefix=OMP-MARCH %s
// OMP-MARCH: "-cc1" "-triple" "amdgpu9.08-amd-amdhsa" {{.*}} "-target-cpu" "gfx908"
// OMP-MARCH-SAME: "-mno-xnack"
// OMP-MARCH-SAME: "-msramecc"

// Test offload with explicit device flags using -Xopenmp-target
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa \
// RUN:   -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx90a \
// RUN:   -Xopenmp-target=amdgcn-amd-amdhsa -mxnack \
// RUN:   -Xopenmp-target=amdgcn-amd-amdhsa -mno-sramecc \
// RUN:   -nogpulib %s 2>&1 | FileCheck -check-prefix=OMP-FLAGS %s
// OMP-FLAGS: "-cc1" "-triple" "amdgpu9.0a-amd-amdhsa" {{.*}} "-target-cpu" "gfx90a"
// OMP-FLAGS-SAME: "-mxnack"
// OMP-FLAGS-SAME: "-mno-sramecc"

// Test offload with target ID taking precedence over explicit flags
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa \
// RUN:   -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx90a:xnack- \
// RUN:   -Xopenmp-target=amdgcn-amd-amdhsa -mxnack \
// RUN:   -nogpulib %s 2>&1 | FileCheck -check-prefix=OMP-TARGETID-WINS %s
// OMP-TARGETID-WINS: "-cc1" "-triple" "amdgpu9.0a-amd-amdhsa" {{.*}} "-target-cpu" "gfx90a"
// OMP-TARGETID-WINS-SAME: "-mno-xnack"

// Test offload using base architecture gfx90a with -mxnack flag for xnack+
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -fopenmp \
// RUN:   --offload-arch=gfx90a \
// RUN:   -Xopenmp-target=amdgcn-amd-amdhsa -mxnack \
// RUN:   -nogpulib %s 2>&1 | FileCheck -check-prefix=OMP-GFX90A-XNACK-ON %s
// OMP-GFX90A-XNACK-ON: "-cc1" "-triple" "amdgpu9.0a-amd-amdhsa" {{.*}} "-target-cpu" "gfx90a"
// OMP-GFX90A-XNACK-ON-SAME: "-mxnack"

// Test offload using base architecture gfx90a with -mno-xnack flag for xnack-
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -fopenmp \
// RUN:   --offload-arch=gfx90a \
// RUN:   -Xopenmp-target=amdgcn-amd-amdhsa -mno-xnack \
// RUN:   -nogpulib %s 2>&1 | FileCheck -check-prefix=OMP-GFX90A-XNACK-OFF %s
// OMP-GFX90A-XNACK-OFF: "-cc1" "-triple" "amdgpu9.0a-amd-amdhsa" {{.*}} "-target-cpu" "gfx90a"
// OMP-GFX90A-XNACK-OFF-SAME: "-mno-xnack"

// Test offload with multiple device compilations for same base architecture.
// To get both xnack+ and xnack- for gfx90a in the same invocation, you must use
// target ID syntax in --offload-arch.
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -fopenmp \
// RUN:   --offload-arch=gfx90a:xnack+ --offload-arch=gfx90a:xnack- -mxnack \
// RUN:   -nogpulib %s 2>&1 | FileCheck -check-prefix=OMP-MULTI-XNACK %s
// OMP-MULTI-XNACK: "-cc1" "-triple" "amdgpu9.0a-amd-amdhsa" {{.*}} "-target-cpu" "gfx90a"
// OMP-MULTI-XNACK-SAME: "-mxnack"
// OMP-MULTI-XNACK: "-cc1" "-triple" "amdgpu9.0a-amd-amdhsa" {{.*}} "-target-cpu" "gfx90a"
// OMP-MULTI-XNACK-SAME: "-mno-xnack"

// Test that -Xopenmp-target flags apply to all targets with matching triple.
// When compiling for multiple different base architectures (gfx906, gfx90a),
// -Xopenmp-target=amdgcn-amd-amdhsa applies the flag to all of them.
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -fopenmp \
// RUN:   --offload-arch=gfx906 --offload-arch=gfx90a \
// RUN:   -Xopenmp-target=amdgcn-amd-amdhsa -mxnack \
// RUN:   -nogpulib %s 2>&1 | FileCheck -check-prefix=OMP-MULTI-ARCH %s
// OMP-MULTI-ARCH: "-cc1" "-triple" "amdgpu9.06-amd-amdhsa" {{.*}} "-target-cpu" "gfx906"
// OMP-MULTI-ARCH-SAME: "-mxnack"
// OMP-MULTI-ARCH: "-cc1" "-triple" "amdgpu9.0a-amd-amdhsa" {{.*}} "-target-cpu" "gfx90a"
// OMP-MULTI-ARCH-SAME: "-mxnack"

// Test that top-level -mxnack flags (not specified to the device are ignored).
// TODO: Should this be forwarded?
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -fopenmp \
// RUN:   --offload-arch=gfx90a -mxnack -mno-sramecc \
// RUN:   -nogpulib %s 2>&1 | FileCheck -check-prefix=GENERIC-ARG %s
// GENERIC-ARG: warning: argument unused during compilation: '-mxnack'
// GENERIC-ARG: warning: argument unused during compilation: '-mno-sramecc'
