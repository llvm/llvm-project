// RUN: clang-tidy -config-file=%S/Inputs/config-file/config-file -dump-config -- | FileCheck %s -check-prefix=CHECK-BASE
// CHECK-BASE: Checks: {{.*}}hicpp-uppercase-literal-suffix
