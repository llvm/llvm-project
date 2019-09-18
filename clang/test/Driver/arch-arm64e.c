// Check that we can manually enable specific ptrauth features.

// RUN: %clang -arch arm64 -c %s -### 2>&1 | FileCheck %s --check-prefix NONE
// NONE: "-cc1"
// NONE-NOT: "-fptrauth-intrinsics"
// NONE-NOT: "-fptrauth-calls"
// NONE-NOT: "-fptrauth-returns"
// NONE-NOT: "-fptrauth-indirect-gotos"
// NONE-NOT: "-fptrauth-auth-traps"

// RUN: %clang -arch arm64 -fptrauth-calls -c %s -### 2>&1 | FileCheck %s --check-prefix CALL
// CALL: "-cc1"{{.*}} {{.*}} "-fptrauth-calls"

// RUN: %clang -arch arm64 -fptrauth-intrinsics -c %s -### 2>&1 | FileCheck %s --check-prefix INTRIN
// INTRIN: "-cc1"{{.*}} {{.*}} "-fptrauth-intrinsics"

// RUN: %clang -arch arm64 -fptrauth-returns -c %s -### 2>&1 | FileCheck %s --check-prefix RETURN
// RETURN: "-cc1"{{.*}} {{.*}} "-fptrauth-returns"

// RUN: %clang -arch arm64 -fptrauth-indirect-gotos -c %s -### 2>&1 | FileCheck %s --check-prefix INDGOTO
// INDGOTO: "-cc1"{{.*}} {{.*}} "-fptrauth-indirect-gotos"

// RUN: %clang -arch arm64 -fptrauth-auth-traps -c %s -### 2>&1 | FileCheck %s --check-prefix TRAPS
// TRAPS: "-cc1"{{.*}} {{.*}} "-fptrauth-auth-traps"


// Check the arm64e defaults.

// RUN: %clang -arch arm64e -c %s -### 2>&1 | FileCheck %s --check-prefix DEFAULT
// RUN: %clang -mkernel -arch arm64e -c %s -### 2>&1 | FileCheck %s --check-prefix DEFAULT
// RUN: %clang -fapple-kext -arch arm64e -c %s -### 2>&1 | FileCheck %s --check-prefix DEFAULT
// DEFAULT: "-fptrauth-returns" "-fptrauth-intrinsics" "-fptrauth-calls" "-fptrauth-indirect-gotos" "-fptrauth-auth-traps" "-target-cpu" "vortex"{{.*}}


// RUN: %clang -arch arm64e -fno-ptrauth-calls -c %s -### 2>&1 | FileCheck %s --check-prefix DEFAULT-NOCALL
// RUN: %clang -mkernel -arch arm64e -fno-ptrauth-calls -c %s -### 2>&1 | FileCheck %s --check-prefix DEFAULT-NOCALL
// RUN: %clang -fapple-kext -arch arm64e -fno-ptrauth-calls -c %s -### 2>&1 | FileCheck %s --check-prefix DEFAULT-NOCALL
// DEFAULT-NOCALL-NOT: "-fptrauth-calls"
// DEFAULT-NOCALL: "-fptrauth-returns" "-fptrauth-intrinsics" "-fptrauth-indirect-gotos" "-fptrauth-auth-traps" "-target-cpu" "vortex"


// RUN: %clang -arch arm64e -fno-ptrauth-returns -c %s -### 2>&1 | FileCheck %s --check-prefix NORET

// NORET-NOT: "-fptrauth-returns"
// NORET: "-fptrauth-intrinsics" "-fptrauth-calls" "-fptrauth-indirect-gotos" "-fptrauth-auth-traps" "-target-cpu" "vortex"

// RUN: %clang -arch arm64e -fno-ptrauth-intrinsics -c %s -### 2>&1 | FileCheck %s --check-prefix NOINTRIN

// NOINTRIN: "-fptrauth-returns"
// NOINTRIN-NOT: "-fptrauth-intrinsics"
// NOINTRIN: "-fptrauth-calls" "-fptrauth-indirect-gotos" "-fptrauth-auth-traps" "-target-cpu" "vortex"{{.*}}


// RUN: %clang -arch arm64e -fno-ptrauth-auth-traps -c %s -### 2>&1 | FileCheck %s --check-prefix NOTRAP
// NOTRAP: "-fptrauth-returns" "-fptrauth-intrinsics" "-fptrauth-calls" "-fptrauth-indirect-gotos" "-target-cpu" "vortex"


// Check the CPU defaults and overrides.

// RUN: %clang -arch arm64e -c %s -### 2>&1 | FileCheck %s --check-prefix VORTEX
// RUN: %clang -arch arm64e -mcpu=vortex -c %s -### 2>&1 | FileCheck %s --check-prefix VORTEX
// RUN: %clang -arch arm64e -mcpu=cyclone -c %s -### 2>&1 | FileCheck %s --check-prefix VORTEX
// RUN: %clang -arch arm64e -mcpu=lightning -c %s -### 2>&1 | FileCheck %s --check-prefix LIGHTNING
// VORTEX: "-cc1"{{.*}} "-target-cpu" "vortex"
// LIGHTNING: "-cc1"{{.*}} "-target-cpu" "lightning"
