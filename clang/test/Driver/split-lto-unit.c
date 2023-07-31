// RUN: %clang --target=x86_64-unknown-linux -### %s -flto=thin 2>&1 | FileCheck --check-prefix=NOUNIT %s
// RUN: %clang --target=x86_64-unknown-linux -### %s -flto=thin -fsplit-lto-unit 2>&1 | FileCheck --check-prefix=UNIT %s
// RUN: %clang --target=x86_64-unknown-linux -### %s -flto=thin -fno-split-lto-unit 2>&1 | FileCheck --check-prefix=NOUNIT %s
// RUN: %clang --target=x86_64-unknown-linux -### %s -flto=thin -fno-split-lto-unit -fwhole-program-vtables 2>&1 | FileCheck --check-prefix=ERROR1 %s
// RUN: %clang --target=x86_64-unknown-linux -### %s -flto=thin -fno-split-lto-unit -fsanitize=cfi 2>&1 | FileCheck --check-prefix=ERROR2 %s
// RUN: %clang --target=x86_64-apple-darwin13.3.0 -### %s -fwhole-program-vtables -flto=full 2>&1 | FileCheck --check-prefix=UNIT %s
// RUN: %clang --target=x86_64-apple-darwin13.3.0 -### %s -fwhole-program-vtables -flto=thin 2>&1 | FileCheck --check-prefix=NOUNIT %s
// RUN: %clang --target=x86_64-scei-ps4 -### %s -fwhole-program-vtables -flto=full 2>&1 | FileCheck --check-prefix=UNIT %s
// RUN: %clang --target=x86_64-scei-ps4 -### %s -fwhole-program-vtables -flto=thin 2>&1 | FileCheck --check-prefix=NOUNIT %s
// RUN: %clang --target=x86_64-sie-ps5 -### %s -fwhole-program-vtables -flto=full 2>&1 | FileCheck --check-prefix=UNIT %s
// RUN: %clang --target=x86_64-sie-ps5 -### %s -fwhole-program-vtables -flto=thin 2>&1 | FileCheck --check-prefix=UNIT %s
// RUN: %clang --target=x86_64-unknown-linux -### %s -fwhole-program-vtables -flto=full -funified-lto 2>&1 | FileCheck --check-prefix=UNIT %s
// RUN: %clang --target=x86_64-unknown-linux -### %s -fwhole-program-vtables -flto=thin -funified-lto 2>&1 | FileCheck --check-prefix=UNIT %s

// UNIT: "-fsplit-lto-unit"
// NOUNIT-NOT: "-fsplit-lto-unit"
// ERROR1-NOT: error: invalid argument
// ERROR2: error: invalid argument '-fno-split-lto-unit' not allowed with '-fsanitize=cfi'
