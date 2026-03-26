// Check that we can manually enable specific ptrauth features.

// RUN: %clang -target arm64-apple-darwin -c %s -### 2>&1 | FileCheck %s --check-prefix NONE
// NONE: "-cc1"

// NONE-NOT: "-fptrauth-calls"
// NONE-NOT: "-fptrauth-returns"
// NONE-NOT: "-fptrauth-intrinsics"
// NONE-NOT: "-fptrauth-indirect-gotos"
// NONE-NOT: "-fptrauth-auth-traps"
// NONE-NOT: "-fptrauth-vtable-pointer-address-discrimination"
// NONE-NOT: "-fptrauth-vtable-pointer-type-discrimination"
// NONE-NOT: "-fptrauth-objc-isa"
// NONE-NOT: "-fptrauth-objc-class-ro"
// NONE-NOT: "-fptrauth-objc-interface-sel"

// Final catch all if any new flags are added
// NONE-NOT: "-fptrauth"

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

// Check the arm64e defaults.

// RUN: %clang -target arm64e-apple-ios -c %s -### 2>&1 | FileCheck %s --check-prefix DEFAULT
// RUN: %clang -target arm64e-apple-macos -c %s -### 2>&1 | FileCheck %s --check-prefix DEFAULTMAC
// RUN: %if system-darwin && target={{.*}}-{{darwin|macos}}{{.*}} %{ %clang -target arm64e-apple-macos -c %s -### 2>&1 | FileCheck %s --check-prefix DEFAULTARCH %}
// RUN: %clang -mkernel -target arm64e-apple-ios -c %s -### 2>&1 | FileCheck %s --check-prefix DEFAULT
// RUN: %clang -fapple-kext -target arm64e-apple-ios -c %s -### 2>&1 | FileCheck %s --check-prefix DEFAULT
// DEFAULT: "-fptrauth-calls" "-fptrauth-returns" "-fptrauth-intrinsics" "-fptrauth-indirect-gotos" "-fptrauth-auth-traps" "-fptrauth-vtable-pointer-address-discrimination" "-fptrauth-vtable-pointer-type-discrimination" "-fptrauth-objc-isa" "-fptrauth-objc-class-ro" "-fptrauth-objc-interface-sel" {{.*}}"-target-cpu" "apple-a12"{{.*}}
// DEFAULTMAC: "-fptrauth-calls" "-fptrauth-returns" "-fptrauth-intrinsics" "-fptrauth-indirect-gotos" "-fptrauth-auth-traps" "-fptrauth-vtable-pointer-address-discrimination" "-fptrauth-vtable-pointer-type-discrimination" "-fptrauth-objc-isa" "-fptrauth-objc-class-ro" "-fptrauth-objc-interface-sel" {{.*}}"-target-cpu" "apple-m1"{{.*}}
// DEFAULTARCH: "-fptrauth-calls" "-fptrauth-returns" "-fptrauth-intrinsics" "-fptrauth-indirect-gotos" "-fptrauth-auth-traps" "-fptrauth-vtable-pointer-address-discrimination" "-fptrauth-vtable-pointer-type-discrimination" "-fptrauth-objc-isa" "-fptrauth-objc-class-ro" "-fptrauth-objc-interface-sel"

// RUN: %clang -target arm64e-apple-none-macho -c %s -### 2>&1 | FileCheck %s --check-prefix DEFAULT-MACHO
// DEFAULT-MACHO: "-fptrauth-calls" "-fptrauth-returns" "-fptrauth-intrinsics" "-fptrauth-indirect-gotos" "-fptrauth-auth-traps" "-fptrauth-vtable-pointer-address-discrimination" "-fptrauth-vtable-pointer-type-discrimination" "-fptrauth-objc-isa" "-fptrauth-objc-class-ro" "-fptrauth-objc-interface-sel" {{.*}}"-target-cpu" "apple-a12"{{.*}}


// RUN: %clang -target arm64e-apple-ios -fno-ptrauth-calls -c %s -### 2>&1 | FileCheck %s --check-prefix DEFAULT-NOCALL
// DEFAULT-NOCALL-NOT: "-fptrauth-calls"
// DEFAULT-NOCALL: "-fptrauth-returns" "-fptrauth-intrinsics" "-fptrauth-indirect-gotos" "-fptrauth-auth-traps" "-fptrauth-vtable-pointer-address-discrimination" "-fptrauth-vtable-pointer-type-discrimination" "-fptrauth-objc-isa" "-fptrauth-objc-class-ro" "-fptrauth-objc-interface-sel" {{.*}}"-target-cpu" "apple-a12"


// RUN: %clang -target arm64e-apple-ios -fno-ptrauth-returns -c %s -### 2>&1 | FileCheck %s --check-prefix NORET

// NORET-NOT: "-fptrauth-returns"
// NORET: "-fptrauth-calls" "-fptrauth-intrinsics" "-fptrauth-indirect-gotos" "-fptrauth-auth-traps" "-fptrauth-vtable-pointer-address-discrimination" "-fptrauth-vtable-pointer-type-discrimination" "-fptrauth-objc-isa" "-fptrauth-objc-class-ro" "-fptrauth-objc-interface-sel" {{.*}}"-target-cpu" "apple-a12"

// RUN: %clang -target arm64e-apple-ios -fno-ptrauth-intrinsics -c %s -### 2>&1 | FileCheck %s --check-prefix NOINTRIN

// NOINTRIN-NOT: "-fptrauth-intrinsics"
// NOINTRIN: "-fptrauth-calls" "-fptrauth-returns" "-fptrauth-indirect-gotos" "-fptrauth-auth-traps" "-fptrauth-vtable-pointer-address-discrimination" "-fptrauth-vtable-pointer-type-discrimination" "-fptrauth-objc-isa" "-fptrauth-objc-class-ro" "-fptrauth-objc-interface-sel" {{.*}}"-target-cpu" "apple-a12"


// RUN: %clang -target arm64e-apple-ios -fno-ptrauth-auth-traps -c %s -### 2>&1 | FileCheck %s --check-prefix NOTRAP
// NOTRAP: "-fptrauth-calls" "-fptrauth-returns" "-fptrauth-intrinsics" "-fptrauth-indirect-gotos" "-fptrauth-vtable-pointer-address-discrimination" "-fptrauth-vtable-pointer-type-discrimination" "-fptrauth-objc-isa" "-fptrauth-objc-class-ro" "-fptrauth-objc-interface-sel" {{.*}}"-target-cpu" "apple-a12"


// Check the CPU defaults and overrides.

// RUN: %clang -target arm64e-apple-ios -c %s -### 2>&1 | FileCheck %s --check-prefix APPLE-A12
// RUN: %clang -target arm64e-apple-ios -mcpu=apple-a13 -c %s -### 2>&1 | FileCheck %s --check-prefix APPLE-A13
// APPLE-A12: "-cc1"{{.*}} "-target-cpu" "apple-a12"
// APPLE-A13: "-cc1"{{.*}} "-target-cpu" "apple-a13"
