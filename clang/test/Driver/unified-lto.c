// RUN: %clang --target=x86_64-unknown-linux -### %s -flto=full -funified-lto 2>&1 | FileCheck --check-prefix=UNIT %s
// RUN: %clang --target=x86_64-unknown-linux -### %s -flto=thin -funified-lto 2>&1 | FileCheck --check-prefix=UNIT %s
// RUN: %clang --target=x86_64-scei-ps4 -### %s -flto=full -funified-lto 2>&1 | FileCheck --check-prefix=UNIT %s
// RUN: %clang --target=x86_64-scei-ps4 -### %s -flto=thin -funified-lto 2>&1 | FileCheck --check-prefix=NOUNIT %s

// UNIT: "-flto-unit"
// NOUNIT-NOT: "-flto-unit"

// RUN: %clang --target=x86_64-sie-prospero -### %s -funified-lto 2>&1 | FileCheck --check-prefix=NOUNILTO %s
// NOUNILTO: clang: warning: argument unused during compilation: '-funified-lto'
// NOUNILTO: "-cc1"
// NOUNILTO-NOT: "-funified-lto
