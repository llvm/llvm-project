// RUN: %clang -target aarch64-linux-gnu -march=armv8-a+sme %s -### 2>&1 | FileCheck %s --check-prefix=SME-IMPLY
// SME-IMPLY: "-target-feature" "+bf16"{{.*}} "-target-feature" "+sme"

// RUN: %clang -target aarch64-linux-gnu -march=armv8-a+nosme %s -### 2>&1 | FileCheck %s --check-prefix=NOSME
// NOSME-NOT: "-target-feature" "{{\+|-}}sme"

// RUN: %clang -target aarch64-linux-gnu -march=armv8-a+sme+nosme %s -### 2>&1 | FileCheck %s --check-prefix=SME-REVERT
// SME-REVERT-NOT: "-target-feature" "+sme"
// SME-REVERT: "-target-feature" "+bf16"{{.*}} "-target-feature" "-sme"

// RUN: %clang -target aarch64-linux-gnu -march=armv8-a+sme+nobf16 %s -### 2>&1 | FileCheck %s --check-prefix=SME-CONFLICT
// SME-CONFLICT-NOT: "-target-feature" "+sme"
// SME-CONFLICT-NOT: "-target-feature" "+bf16"
// SME-CONFLICT: "-target-feature" "-bf16"{{.*}} "-target-feature" "-sme"

// RUN: %clang -target aarch64-linux-gnu -march=armv8-a+sme-i16i64 %s -### 2>&1 | FileCheck %s --check-prefix=SME-I16I64
// SME-I16I64: "-target-feature" "+bf16"{{.*}} "-target-feature" "+sme" "-target-feature" "+sme-i16i64"

// RUN: %clang -target aarch64-linux-gnu -march=armv8-a+nosme-i16i64 %s -### 2>&1 | FileCheck %s --check-prefix=NOSME-I16I64
// NOSME-I16I64-NOT: "-target-feature" "+sme-i16i64"
// NOSME-I16I64-NOT: "-target-feature" "+sme"
// NOSME-I16I64-NOT: "-target-feature" "+bf16"
// NOSME-I16I64-NOT: sme-i16i64"

// RUN: %clang -target aarch64-linux-gnu -march=armv8-a+sme-i16i64+nosme-i16i64 %s -### 2>&1 | FileCheck %s --check-prefix=SME-I16I64-REVERT
// SME-I16I64-REVERT: "-target-feature" "+bf16"{{.*}} "-target-feature" "+sme" "-target-feature" "-sme-i16i64"

// RUN: %clang -target aarch64-linux-gnu -march=armv8-a+nosme-f64f64 %s -### 2>&1 | FileCheck %s --check-prefix=NOSME-F64F64
// NOSME-F64F64-NOT: "-target-feature" "+sme-f64f64"
// NOSME-F64F64-NOT: "-target-feature" "+sme"
// NOSME-F64F64-NOT: "-target-feature" "+bf16"
// NOSME-F64F64-NOT: sme-f64f64"

// RUN: %clang -target aarch64-linux-gnu -march=armv8-a+sme-f64f64+nosme-f64f64 %s -### 2>&1 | FileCheck %s --check-prefix=SME-F64F64-REVERT
// SME-F64F64-REVERT: "-target-feature" "+bf16"{{.*}} "-target-feature" "+sme" "-target-feature" "-sme-f64f64"

// RUN: %clang -target aarch64-linux-gnu -march=armv8-a+sme-f64f64+nosme-i16i64 %s -### 2>&1 | FileCheck %s --check-prefix=SME-SUBFEATURE-MIX
// SME-SUBFEATURE-MIX-NOT: "+sme-i16i64"
// SME-SUBFEATURE-MIX: "-target-feature" "+bf16"{{.*}} "-target-feature" "+sme" "-target-feature" "+sme-f64f64"
// SME-SUBFEATURE-MIX-NOT: "+sme-i16i64"

// RUN: %clang -target aarch64-linux-gnu -march=armv8-a+sme-i16i64+nosme %s -### 2>&1 | FileCheck %s --check-prefix=SME-SUBFEATURE-CONFLICT1
// SME-SUBFEATURE-CONFLICT1: "-target-feature" "+bf16"{{.*}} "-target-feature" "-sme" "-target-feature" "-sme-i16i64"

// RUN: %clang -target aarch64-linux-gnu -march=armv8-a+sme-f64f64+nobf16 %s -### 2>&1 | FileCheck %s --check-prefix=SME-SUBFEATURE-CONFLICT2
// SME-SUBFEATURE-CONFLICT2-NOT: "-target-feature" "+bf16"
// SME-SUBFEATURE-CONFLICT2-NOT: "-target-feature" "+sme"
// SME-SUBFEATURE-CONFLICT2-NOT: "-target-feature" "+sme-f64f64"

// RUN: %clang -target aarch64-linux-gnu -march=armv8-a+nosme+sme-i16i64 %s -### 2>&1 | FileCheck %s --check-prefix=SME-SUBFEATURE-CONFLICT-REV
// SME-SUBFEATURE-CONFLICT-REV: "-target-feature" "+bf16"{{.*}} "-target-feature" "+sme" "-target-feature" "+sme-i16i64"

// RUN: %clang --target=aarch64-linux-gnu -march=armv8-a+ssve-aes %s -### 2>&1 | FileCheck %s --check-prefix=SVE-AES
// SVE-AES: "-target-feature" "+sme" "-target-feature" "+sme2" "-target-feature" "+ssve-aes" "-target-feature" "+sve-aes"