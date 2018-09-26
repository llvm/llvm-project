// RUN: %clang -target aarch64-none-gnu -fcall-saved-x8 -### %s  2>&1  \
// RUN: | FileCheck --check-prefix=CHECK-CALL-SAVED-X8 %s

// RUN: %clang -target aarch64-none-gnu -fcall-saved-x9 -### %s  2>&1  \
// RUN: | FileCheck --check-prefix=CHECK-CALL-SAVED-X9 %s

// RUN: %clang -target aarch64-none-gnu -fcall-saved-x10 -### %s  2>&1  \
// RUN: | FileCheck --check-prefix=CHECK-CALL-SAVED-X10 %s

// RUN: %clang -target aarch64-none-gnu -fcall-saved-x11 -### %s  2>&1  \
// RUN: | FileCheck --check-prefix=CHECK-CALL-SAVED-X11 %s

// RUN: %clang -target aarch64-none-gnu -fcall-saved-x12 -### %s  2>&1  \
// RUN: | FileCheck --check-prefix=CHECK-CALL-SAVED-X12 %s

// RUN: %clang -target aarch64-none-gnu -fcall-saved-x13 -### %s  2>&1  \
// RUN: | FileCheck --check-prefix=CHECK-CALL-SAVED-X13 %s

// RUN: %clang -target aarch64-none-gnu -fcall-saved-x14 -### %s  2>&1  \
// RUN: | FileCheck --check-prefix=CHECK-CALL-SAVED-X14 %s

// RUN: %clang -target aarch64-none-gnu -fcall-saved-x15 -### %s  2>&1  \
// RUN: | FileCheck --check-prefix=CHECK-CALL-SAVED-X15 %s

// RUN: %clang -target aarch64-none-gnu -fcall-saved-x18 -### %s  2>&1  \
// RUN: | FileCheck --check-prefix=CHECK-CALL-SAVED-X18 %s

// Test all call-saved-x# options together.
// RUN: %clang -target aarch64-none-gnu \
// RUN: -fcall-saved-x8 \
// RUN: -fcall-saved-x9 \
// RUN: -fcall-saved-x10 \
// RUN: -fcall-saved-x11 \
// RUN: -fcall-saved-x12 \
// RUN: -fcall-saved-x13 \
// RUN: -fcall-saved-x14 \
// RUN: -fcall-saved-x15 \
// RUN: -fcall-saved-x18 \
// RUN: -### %s  2>&1 | FileCheck %s \
// RUN: --check-prefix=CHECK-CALL-SAVED-X8 \
// RUN: --check-prefix=CHECK-CALL-SAVED-X9 \
// RUN: --check-prefix=CHECK-CALL-SAVED-X10 \
// RUN: --check-prefix=CHECK-CALL-SAVED-X11 \
// RUN: --check-prefix=CHECK-CALL-SAVED-X12 \
// RUN: --check-prefix=CHECK-CALL-SAVED-X13 \
// RUN: --check-prefix=CHECK-CALL-SAVED-X14 \
// RUN: --check-prefix=CHECK-CALL-SAVED-X15 \
// RUN: --check-prefix=CHECK-CALL-SAVED-X18

// CHECK-CALL-SAVED-X8: "-target-feature" "+call-saved-x8"
// CHECK-CALL-SAVED-X9: "-target-feature" "+call-saved-x9"
// CHECK-CALL-SAVED-X10: "-target-feature" "+call-saved-x10"
// CHECK-CALL-SAVED-X11: "-target-feature" "+call-saved-x11"
// CHECK-CALL-SAVED-X12: "-target-feature" "+call-saved-x12"
// CHECK-CALL-SAVED-X13: "-target-feature" "+call-saved-x13"
// CHECK-CALL-SAVED-X14: "-target-feature" "+call-saved-x14"
// CHECK-CALL-SAVED-X15: "-target-feature" "+call-saved-x15"
// CHECK-CALL-SAVED-X18: "-target-feature" "+call-saved-x18"
