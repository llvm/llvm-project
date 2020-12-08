// RUN: %clang -### -target amdgcn-amd-amdhsa -mcpu=gfx700 -mcode-object-v3 %s 2>&1 | FileCheck --check-prefix=CODE-OBJECT-V3 %s
// CODE-OBJECT-V3: warning: argument '-mcode-object-v3' is deprecated, use '-mcode-object-version=3' instead [-Wdeprecated]
// CODE-OBJECT-V3: "-mllvm" "--amdhsa-code-object-version=3"

// RUN: %clang -### -target amdgcn-amd-amdhsa amdgcn -mcpu=gfx700 -mno-code-object-v3 %s 2>&1 | FileCheck --check-prefix=NO-CODE-OBJECT-V3 %s
// NO-CODE-OBJECT-V3: warning: argument '-mno-code-object-v3' is deprecated, use '-mcode-object-version=2' instead [-Wdeprecated]
// NO-CODE-OBJECT-V3: "-mllvm" "--amdhsa-code-object-version=2"

// RUN: %clang -### -target amdgcn-amd-amdhsa -mcpu=gfx700 -mcode-object-v3 -mno-code-object-v3 -mcode-object-v3 %s 2>&1 | FileCheck --check-prefix=MUL-CODE-OBJECT-V3 %s
// MUL-CODE-OBJECT-V3: warning: argument '-mcode-object-v3' is deprecated, use '-mcode-object-version=3' instead [-Wdeprecated]
// MUL-CODE-OBJECT-V3: "-mllvm" "--amdhsa-code-object-version=3"

// RUN: %clang -### -target amdgcn-amdhsa -mcpu=gfx900:xnack+ %s 2>&1 | FileCheck --check-prefix=XNACK %s
// XNACK: "-target-feature" "+xnack"

// RUN: %clang -### -target amdgcn-amdpal -mcpu=gfx900:xnack- %s 2>&1 | FileCheck --check-prefix=NO-XNACK %s
// NO-XNACK: "-target-feature" "-xnack"

// RUN: %clang -### -target amdgcn-mesa3d -mcpu=gfx908:sramecc+ %s 2>&1 | FileCheck --check-prefix=SRAM-ECC %s
// SRAM-ECC: "-target-feature" "+sramecc"

// RUN: %clang -### -target amdgcn-amdhsa -mcpu=gfx908:sramecc- %s 2>&1 | FileCheck --check-prefix=NO-SRAM-ECC %s
// NO-SRAM-ECC: "-target-feature" "-sramecc"

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
