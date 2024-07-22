// RUN: %clang -### --target=amdgcn-amdhsa -mcpu=gfx900:xnack+ -nogpulib %s 2>&1 | FileCheck --check-prefix=XNACK %s
// XNACK: "-target-feature" "+xnack"

// RUN: %clang -### -target amdgcn-amdpal -mcpu=gfx900:xnack- %s 2>&1 | FileCheck --check-prefix=NO-XNACK %s
// NO-XNACK: "-target-feature" "-xnack"

// RUN: %clang -### -target amdgcn-mesa3d -mcpu=gfx908:sramecc+ %s 2>&1 | FileCheck --check-prefix=SRAM-ECC %s
// SRAM-ECC: "-target-feature" "+sramecc"

// RUN: %clang -### --target=amdgcn-amdhsa -mcpu=gfx908:sramecc- -nogpulib %s 2>&1 | FileCheck --check-prefix=NO-SRAM-ECC %s
// NO-SRAM-ECC: "-target-feature" "-sramecc"

// RUN: %clang -### -target amdgcn -mcpu=gfx90A -mtgsplit %s 2>&1 | FileCheck --check-prefix=TGSPLIT %s
// RUN: %clang -### -target amdgcn -mcpu=gfx90A -mno-tgsplit %s 2>&1 | FileCheck --check-prefix=NO-TGSPLIT %s
// TGSPLIT: "-target-feature" "+tgsplit"
// NO-TGSPLIT: "-target-feature" "-tgsplit"

// RUN: %clang -### -target amdgcn-amdpal -mcpu=gfx1010 -mwavefrontsize64 %s 2>&1 | FileCheck --check-prefix=WAVE64 %s
// RUN: %clang -### -target amdgcn-amdpal -mcpu=gfx1010 -mno-wavefrontsize64 -mwavefrontsize64 %s 2>&1 | FileCheck --check-prefix=WAVE64 %s
// WAVE64: "-target-feature" "+wavefrontsize64"
// WAVE64-NOT: {{".*wavefrontsize16"}}
// WAVE64-NOT: {{".*wavefrontsize32"}}

// RUN: %clang -### -target amdgcn -mcpu=gfx1010 -mno-wavefrontsize64 %s 2>&1 | FileCheck --check-prefix=NO-WAVE64 %s
// RUN: %clang -### -target amdgcn -mcpu=gfx1010 -mwavefrontsize64 -mno-wavefrontsize64 %s 2>&1 | FileCheck --check-prefix=NO-WAVE64 %s
// NO-WAVE64-NOT: {{".*wavefrontsize16"}}
// NO-WAVE64-NOT: {{".*wavefrontsize32"}}
// NO-WAVE64-NOT: {{".*wavefrontsize64"}}

// RUN: %clang -### -target amdgcn -mcpu=gfx1010 -mcumode %s 2>&1 | FileCheck --check-prefix=CUMODE %s
// CUMODE: "-target-feature" "+cumode"

// RUN: %clang -### -target amdgcn -mcpu=gfx1010 -mno-cumode %s 2>&1 | FileCheck --check-prefix=NO-CUMODE %s
// NO-CUMODE: "-target-feature" "-cumode"

// RUN: %clang -### -target amdgcn -mcpu=gfx1010 -mamdgpu-precise-memory-op %s 2>&1 | FileCheck --check-prefix=PREC-MEM %s
// PREC-MEM: "-target-feature" "+precise-memory"

// RUN: %clang -### -target amdgcn -mcpu=gfx1010 -mno-amdgpu-precise-memory-op %s 2>&1 | FileCheck --check-prefix=NO-PREC-MEM %s
// NO-PREC-MEM-NOT: {{".*precise-memory"}}
