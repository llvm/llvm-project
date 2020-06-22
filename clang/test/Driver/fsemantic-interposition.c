// RUN: %clang -target x86_64 %s -Werror -fpic -fsemantic-interposition -c -### 2>&1 | FileCheck %s
// RUN: %clang -target x86_64 %s -Werror -fPIC -fsemantic-interposition -c -### 2>&1 | FileCheck %s
// CHECK: "-fsemantic-interposition"

/// Require explicit -fno-semantic-interposition to infer dso_local.
// RUN: %clang -target x86_64 %s -Werror -fPIC -fsemantic-interposition -fno-semantic-interposition -c -### 2>&1 | FileCheck --check-prefix=EXPLICIT_NO %s
// EXPLICIT_NO: "-fno-semantic-interposition"

// RUN: %clang -target x86_64 %s -Werror -fsemantic-interposition -c -### 2>&1 | FileCheck --check-prefix=NO %s
// RUN: %clang -target x86_64 %s -Werror -fPIC -c -### 2>&1 | FileCheck --check-prefix=NO %s
// RUN: %clang -target x86_64 %s -Werror -fPIE -fsemantic-interposition -c -### 2>&1 | FileCheck --check-prefix=NO %s
// NO-NOT: "-fsemantic-interposition"
// NO-NOT: "-fno-semantic-interposition"
