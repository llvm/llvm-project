// RUN: %clang -target armv7-unknown-unknown -### -fsyntax-only %s 2> %t \
// RUN:   -mno-global-merge
// RUN: FileCheck --check-prefix=CHECK-NGM-ARM < %t %s

// RUN: %clang -target aarch64-unknown-unknown -### -fsyntax-only %s 2> %t \
// RUN:   -mno-global-merge
// RUN: FileCheck --check-prefix=CHECK-NGM-AARCH64 < %t %s

// RUN: %clang -target x86_64-unknown-unknown -### -fsyntax-only %s 2> %t \
// RUN:   -mno-global-merge
// RUN: FileCheck --check-prefix=CHECK-NONE < %t %s

// CHECK-NGM-ARM: "-mllvm" "-arm-global-merge=false"
// CHECK-NGM-AARCH64: "-mllvm" "-aarch64-enable-global-merge=false"

// RUN: %clang -target armv7-unknown-unknown -### -fsyntax-only %s 2> %t \
// RUN:   -mglobal-merge
// RUN: FileCheck --check-prefix=CHECK-GM-ARM < %t %s

// RUN: %clang -target aarch64-unknown-unknown -### -fsyntax-only %s 2> %t \
// RUN:   -mglobal-merge
// RUN: FileCheck --check-prefix=CHECK-GM-AARCH64 < %t %s

// RUN: %clang -target x86_64-unknown-unknown -### -fsyntax-only %s 2> %t \
// RUN:   -mglobal-merge
// RUN: FileCheck --check-prefix=CHECK-NONE < %t %s

// CHECK-GM-ARM: "-mllvm" "-arm-global-merge=true"
// CHECK-GM-AARCH64: "-mllvm" "-aarch64-enable-global-merge=true"

// RUN: %clang -target armv7-unknown-unknown -### -fsyntax-only %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-NONE < %t %s

// RUN: %clang -target aarch64-unknown-unknown -### -fsyntax-only %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-NONE < %t %s

// RUN: %clang -target x86_64-unknown-unknown -### -fsyntax-only %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-NONE < %t %s

// CHECK-NONE-NOT: -global-merge=
