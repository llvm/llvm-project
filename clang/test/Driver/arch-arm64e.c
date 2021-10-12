// Check that we can manually enable specific ptrauth features.

// RUN: %clang -target arm64-apple-darwin -c %s -### 2>&1 | FileCheck %s --check-prefix NONE
// NONE: "-cc1"
// NONE-NOT: "-fptrauth-intrinsics"
// NONE-NOT: "-fptrauth-calls"
// NONE-NOT: "-fptrauth-returns"
// NONE-NOT: "-fptrauth-indirect-gotos"
// NONE-NOT: "-fptrauth-auth-traps"
// NONE-NOT: "-fptrauth-soft"

// RUN: %clang -target arm64-apple-darwin -fptrauth-calls -c %s -### 2>&1 | FileCheck %s --check-prefix CALL
// CALL: "-cc1"{{.*}} {{.*}} "-fptrauth-calls"

// RUN: %clang -target arm64-apple-darwin -fptrauth-intrinsics -c %s -### 2>&1 | FileCheck %s --check-prefix INTRIN
// INTRIN: "-cc1"{{.*}} {{.*}} "-fptrauth-intrinsics"

// RUN: %clang -target arm64-apple-darwin -fptrauth-returns -c %s -### 2>&1 | FileCheck %s --check-prefix RETURN
// RETURN: "-cc1"{{.*}} {{.*}} "-fptrauth-returns"

// RUN: %clang -target arm64-apple-darwin -fptrauth-indirect-gotos -c %s -### 2>&1 | FileCheck %s --check-prefix INDGOTO
// INDGOTO: "-cc1"{{.*}} {{.*}} "-fptrauth-indirect-gotos"

// RUN: %clang -target arm64-apple-darwin -fptrauth-auth-traps -c %s -### 2>&1 | FileCheck %s --check-prefix TRAPS
// TRAPS: "-cc1"{{.*}} {{.*}} "-fptrauth-auth-traps"

// RUN: %clang -target arm64-apple-darwin -fptrauth-soft -c %s -### 2>&1 | FileCheck %s --check-prefix SOFT
// SOFT: "-cc1"{{.*}} {{.*}} "-fptrauth-soft"


// Check the arm64e defaults.

// RUN: %clang -target arm64e-apple-darwin -c %s -### 2>&1 | FileCheck %s --check-prefix DEFAULT
// RUN: %clang -mkernel -target arm64e-apple-darwin -c %s -### 2>&1 | FileCheck %s --check-prefix DEFAULT
// RUN: %clang -fapple-kext -target arm64e-apple-darwin -c %s -### 2>&1 | FileCheck %s --check-prefix DEFAULT
// DEFAULT: "-fptrauth-returns" "-fptrauth-intrinsics" "-fptrauth-calls" "-fptrauth-indirect-gotos" "-fptrauth-auth-traps" "-target-cpu" "apple-a12"{{.*}}


// RUN: %clang -target arm64e-apple-darwin -fno-ptrauth-calls -c %s -### 2>&1 | FileCheck %s --check-prefix DEFAULT-NOCALL
// RUN: %clang -mkernel -target arm64e-apple-darwin -fno-ptrauth-calls -c %s -### 2>&1 | FileCheck %s --check-prefix DEFAULT-NOCALL
// RUN: %clang -fapple-kext -target arm64e-apple-darwin -fno-ptrauth-calls -c %s -### 2>&1 | FileCheck %s --check-prefix DEFAULT-NOCALL
// DEFAULT-NOCALL-NOT: "-fptrauth-calls"
// DEFAULT-NOCALL: "-fptrauth-returns" "-fptrauth-intrinsics" "-fptrauth-indirect-gotos" "-fptrauth-auth-traps" "-target-cpu" "apple-a12"


// RUN: %clang -target arm64e-apple-darwin -fno-ptrauth-returns -c %s -### 2>&1 | FileCheck %s --check-prefix NORET

// NORET-NOT: "-fptrauth-returns"
// NORET: "-fptrauth-intrinsics" "-fptrauth-calls" "-fptrauth-indirect-gotos" "-fptrauth-auth-traps" "-target-cpu" "apple-a12"

// RUN: %clang -target arm64e-apple-darwin -fno-ptrauth-intrinsics -c %s -### 2>&1 | FileCheck %s --check-prefix NOINTRIN

// NOINTRIN: "-fptrauth-returns"
// NOINTRIN-NOT: "-fptrauth-intrinsics"
// NOINTRIN: "-fptrauth-calls" "-fptrauth-indirect-gotos" "-fptrauth-auth-traps" "-target-cpu" "apple-a12"{{.*}}


// RUN: %clang -target arm64e-apple-darwin -fno-ptrauth-auth-traps -c %s -### 2>&1 | FileCheck %s --check-prefix NOTRAP
// NOTRAP: "-fptrauth-returns" "-fptrauth-intrinsics" "-fptrauth-calls" "-fptrauth-indirect-gotos" "-target-cpu" "apple-a12"


// Check the CPU defaults and overrides.

// RUN: %clang -target arm64e-apple-darwin -c %s -### 2>&1 | FileCheck %s --check-prefix APPLE-A12
// RUN: %clang -target arm64e-apple-darwin -mcpu=apple-a13 -c %s -### 2>&1 | FileCheck %s --check-prefix APPLE-A13
// APPLE-A12: "-cc1"{{.*}} "-target-cpu" "apple-a12"
// APPLE-A13: "-cc1"{{.*}} "-target-cpu" "apple-a13"
