// RUN: clang-tidy -config-file=%S/Inputs/config-file/config-file -dump-config -- | FileCheck %s -check-prefix=CHECK-BASE
// CHECK-BASE: Checks: {{.*}}hicpp-uppercase-literal-suffix
// RUN: clang-tidy -config-file=%S/Inputs/config-file/config-file-spaces --list-checks -- | FileCheck %s -check-prefix=CHECK-SPACES
// CHECK-SPACES: Enabled checks:
// CHECK-SPACES-NEXT: hicpp-uppercase-literal-suffix
// CHECK-SPACES-NEXT: hicpp-use-auto
// CHECK-SPACES-NEXT: hicpp-use-emplace
// CHECK-SPACES-EMPTY:
// RUN: clang-tidy -config-file=%S/Inputs/config-file/config-file-list-dash --list-checks -- | FileCheck %s -check-prefix=CHECK-LIST-DASH
// CHECK-LIST-DASH: Enabled checks:
// CHECK-LIST-DASH-NEXT: hicpp-uppercase-literal-suffix
// CHECK-LIST-DASH-NEXT: hicpp-use-auto
// CHECK-LIST-DASH-NEXT: hicpp-use-emplace
// CHECK-LIST-DASH-EMPTY:
// RUN: clang-tidy -config-file=%S/Inputs/config-file/config-file-list-bracket --list-checks -- | FileCheck %s -check-prefix=CHECK-LIST-BRACKET
// CHECK-LIST-BRACKET: Enabled checks:
// CHECK-LIST-BRACKET-NEXT: hicpp-uppercase-literal-suffix
// CHECK-LIST-BRACKET-NEXT: hicpp-use-auto
// CHECK-LIST-BRACKET-NEXT: hicpp-use-emplace
// CHECK-LIST-BRACKET-EMPTY:
