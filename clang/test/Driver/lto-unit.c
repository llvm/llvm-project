// RUN: %clang -target x86_64-unknown-linux -### %s -flto=full 2>&1 | FileCheck --check-prefix=UNIT %s
// RUN: %clang -target x86_64-unknown-linux -### %s -flto=thin 2>&1 | FileCheck --check-prefix=UNIT %s
// RUN: %clang -target x86_64-apple-darwin13.3.0 -### %s -flto=full 2>&1 | FileCheck --check-prefix=UNIT %s
// RUN: %clang -target x86_64-apple-darwin13.3.0 -### %s -flto=thin 2>&1 | FileCheck --check-prefix=NOUNIT %s
// RUN: %clang -target x86_64-scei-ps4 -### %s -flto=full 2>&1 | FileCheck --check-prefix=UNIT %s
// RUN: %clang -target x86_64-scei-ps4 -### %s -flto=thin 2>&1 | FileCheck --check-prefix=NOUNIT %s

// UNIT: "-flto-unit"
// NOUNIT-NOT: "-flto-unit"
