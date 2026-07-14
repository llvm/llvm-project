// RUN: clang-tidy -config-file=%S/Inputs/config-file/config-file -dump-config -- | FileCheck %s -check-prefix=CHECK-BASE
// CHECK-BASE: Checks: {{.*}}modernize-use-nullptr
// RUN: clang-tidy -config-file=%S/Inputs/config-file/config-file-spaces --list-checks -- | FileCheck %s -check-prefix=CHECK-SPACES
// CHECK-SPACES: Enabled checks:
// CHECK-SPACES-NEXT: modernize-use-auto
// CHECK-SPACES-NEXT: modernize-use-emplace
// CHECK-SPACES-NEXT: modernize-use-nullptr
// CHECK-SPACES-EMPTY:
// RUN: clang-tidy -config-file=%S/Inputs/config-file/config-file-list-dash --list-checks -- | FileCheck %s -check-prefix=CHECK-LIST-DASH
// CHECK-LIST-DASH: Enabled checks:
// CHECK-LIST-DASH-NEXT: modernize-use-auto
// CHECK-LIST-DASH-NEXT: modernize-use-emplace
// CHECK-LIST-DASH-NEXT: modernize-use-nullptr
// CHECK-LIST-DASH-EMPTY:
// RUN: clang-tidy -config-file=%S/Inputs/config-file/config-file-list-bracket --list-checks -- | FileCheck %s -check-prefix=CHECK-LIST-BRACKET
// CHECK-LIST-BRACKET: Enabled checks:
// CHECK-LIST-BRACKET-NEXT: modernize-use-auto
// CHECK-LIST-BRACKET-NEXT: modernize-use-emplace
// CHECK-LIST-BRACKET-NEXT: modernize-use-nullptr
// CHECK-LIST-BRACKET-EMPTY:
