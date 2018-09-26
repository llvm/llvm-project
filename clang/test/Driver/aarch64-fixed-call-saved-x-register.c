// Check that -ffixed and -fcall-saved flags work correctly together.
// RUN: %clang -target aarch64-none-gnu \
// RUN: -ffixed-x18 \
// RUN: -fcall-saved-x18 \
// RUN: -### %s  2>&1 | FileCheck %s

// CHECK: "-target-feature" "+reserve-x18"
// CHECK: "-target-feature" "+call-saved-x18"
