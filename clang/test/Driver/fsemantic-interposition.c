// RUN: %clang -target x86_64 %s -Werror -fpic -fsemantic-interposition -c -### 2>&1 | FileCheck %s
// RUN: %clang -target x86_64 %s -Werror -fPIC -fsemantic-interposition -c -### 2>&1 | FileCheck %s
// CHECK: "-fsemantic-interposition"

// RUN: %clang -target x86_64 %s -Werror -fPIC -fsemantic-interposition -fno-semantic-interposition -c -### 2>&1 | FileCheck --check-prefix=NO %s
// RUN: %clang -target x86_64 %s -Werror -fsemantic-interposition -c -### 2>&1 | FileCheck --check-prefix=NO %s
// RUN: %clang -target x86_64 %s -Werror -fPIC -c -### 2>&1 | FileCheck --check-prefix=NO %s
// RUN: %clang -target x86_64 %s -Werror -fPIE -fsemantic-interposition -c -### 2>&1 | FileCheck --check-prefix=NO %s
// NO-NOT: "-fsemantic-interposition"
