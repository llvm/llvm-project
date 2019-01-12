// RUN: %clang -target x86_64-unknown-linux -### %s -flto=thin 2>&1 | FileCheck --check-prefix=NOUNIT %s
// RUN: %clang -target x86_64-unknown-linux -### %s -flto=thin -fsplit-lto-unit 2>&1 | FileCheck --check-prefix=UNIT %s
// RUN: %clang -target x86_64-unknown-linux -### %s -flto=thin -fno-split-lto-unit 2>&1 | FileCheck --check-prefix=NOUNIT %s
// RUN: %clang -target x86_64-unknown-linux -### %s -flto=thin -fno-split-lto-unit -fwhole-program-vtables 2>&1 | FileCheck --check-prefix=ERROR1 %s
// RUN: %clang -target x86_64-unknown-linux -### %s -flto=thin -fno-split-lto-unit -fsanitize=cfi 2>&1 | FileCheck --check-prefix=ERROR2 %s

// UNIT: "-fsplit-lto-unit"
// NOUNIT-NOT: "-fsplit-lto-unit"
// ERROR1: error: invalid argument '-fno-split-lto-unit' not allowed with '-fwhole-program-vtables'
// ERROR2: error: invalid argument '-fno-split-lto-unit' not allowed with '-fsanitize=cfi'
