// RUN: %clang --target=sparc -mfpu %s -### 2>&1 | FileCheck -check-prefix=FPU %s
// RUN: %clang --target=sparc -mno-fpu %s -### 2>&1 | FileCheck -check-prefix=NO-FPU %s
// FPU: "-mfloat-abi" "hard"
// NO-FPU: "-mfloat-abi" "soft"

// RUN: %clang --target=sparc -mfsmuld %s -### 2>&1 | FileCheck -check-prefix=FSMULD %s
// RUN: %clang --target=sparc -mno-fsmuld %s -### 2>&1 | FileCheck -check-prefix=NO-FSMULD %s
// FSMULD: "-target-feature" "+fsmuld"
// NO-FSMULD: "-target-feature" "-fsmuld"

// RUN: %clang --target=sparc -mpopc %s -### 2>&1 | FileCheck -check-prefix=POPC %s
// RUN: %clang --target=sparc -mno-popc %s -### 2>&1 | FileCheck -check-prefix=NO-POPC %s
// POPC: "-target-feature" "+popc"
// NO-POPC: "-target-feature" "-popc"

// RUN: %clang --target=sparc -mvis %s -### 2>&1 | FileCheck -check-prefix=VIS %s
// RUN: %clang --target=sparc -mno-vis %s -### 2>&1 | FileCheck -check-prefix=NO-VIS %s
// VIS: "-target-feature" "+vis"
// NO-VIS: "-target-feature" "-vis"

// RUN: %clang --target=sparc -mvis2 %s -### 2>&1 | FileCheck -check-prefix=VIS2 %s
// RUN: %clang --target=sparc -mno-vis2 %s -### 2>&1 | FileCheck -check-prefix=NO-VIS2 %s
// VIS2: "-target-feature" "+vis2"
// NO-VIS2: "-target-feature" "-vis2"

// RUN: %clang --target=sparc -mvis3 %s -### 2>&1 | FileCheck -check-prefix=VIS3 %s
// RUN: %clang --target=sparc -mno-vis3 %s -### 2>&1 | FileCheck -check-prefix=NO-VIS3 %s
// VIS3: "-target-feature" "+vis3"
// NO-VIS3: "-target-feature" "-vis3"

// RUN: %clang --target=sparc -mhard-quad-float %s -### 2>&1 | FileCheck -check-prefix=HARD-QUAD-FLOAT %s
// RUN: %clang --target=sparc -msoft-quad-float %s -### 2>&1 | FileCheck -check-prefix=SOFT-QUAD-FLOAT %s
// HARD-QUAD-FLOAT: "-target-feature" "+hard-quad-float"
// SOFT-QUAD-FLOAT: "-target-feature" "-hard-quad-float"

// RUN: %clang --target=sparc -mv8plus %s -### 2>&1 | FileCheck -check-prefix=V8PLUS %s
// V8PLUS: "-target-feature" "+v8plus"
