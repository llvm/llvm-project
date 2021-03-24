// RUN: clang-tidy -config-file=%S/Inputs/config-file/config-file -dump-config -- | FileCheck %s -check-prefix=CHECK-BASE
// CHECK-BASE: Checks: {{.*}}hicpp-uppercase-literal-suffix
// RUN: clang-tidy -config-file=%S/Inputs/config-file/config-file-spaces --list-checks -- | FileCheck %s -check-prefix=CHECK-SPACES
// CHECK-SPACES: Enabled checks:
// CHECK-SPACES-NEXT: hicpp-uppercase-literal-suffix
// CHECK-SPACES-NEXT: hicpp-use-auto
// CHECK-SPACES-NEXT: hicpp-use-emplace
// CHECK-SPACES-EMPTY:
