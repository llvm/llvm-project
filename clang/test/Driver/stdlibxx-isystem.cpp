// Backslash escaping makes matching against the installation directory fail on
// Windows. Temporarily disable the test there until we add an option to print
// the installation directory unescaped.
// UNSUPPORTED: system-windows

// By default, we should search for libc++ next to the driver.
// RUN: mkdir -p %t/bin
// RUN: mkdir -p %t/include/c++/v1
// RUN: %clang -target aarch64-linux-gnu -ccc-install-dir %t/bin \
// RUN:   -stdlib=libc++ -fsyntax-only %s -### 2>&1 | \
// RUN:   FileCheck -check-prefix=LIBCXX %s
// RUN: %clang -target x86_64-apple-darwin -ccc-install-dir %t/bin \
// RUN:   -stdlib=libc++ -fsyntax-only %s -### 2>&1 | \
// RUN:   FileCheck -check-prefix=LIBCXX %s
// LIBCXX: InstalledDir: [[INSTALLDIR:.+$]]
// LIBCXX: "-internal-isystem" "[[INSTALLDIR]]/../include/c++/v1"

// Passing -stdlib++-isystem should suppress the default search.
// RUN: %clang -target aarch64-linux-gnu -ccc-install-dir %t/bin \
// RUN:   -stdlib++-isystem /tmp/foo -stdlib++-isystem /tmp/bar -stdlib=libc++ \
// RUN:   -fsyntax-only %s -### 2>&1 | FileCheck -check-prefix=NODEFAULT %s
// RUN: %clang -target x86_64-apple-darwin -ccc-install-dir %t/bin \
// RUN:   -stdlib++-isystem /tmp/foo -stdlib++-isystem /tmp/bar -stdlib=libc++ \
// RUN:   -fsyntax-only %s -### 2>&1 | FileCheck -check-prefix=NODEFAULT %s
// NODEFAULT: InstalledDir: [[INSTALLDIR:.+$]]
// NODEFAULT-NOT: "-internal-isystem" "[[INSTALLDIR]]/../include/c++/v1"

// And we should add it as an -internal-isystem.
// RUN: %clang -target aarch64-linux-gnu -ccc-install-dir %t/bin \
// RUN:   -stdlib++-isystem /tmp/foo -stdlib++-isystem /tmp/bar -stdlib=libc++ \
// RUN:   -fsyntax-only %s -### 2>&1 | FileCheck -check-prefix=INCPATH %s
// RUN: %clang -target x86_64-apple-darwin -ccc-install-dir %t/bin \
// RUN:   -stdlib++-isystem /tmp/foo -stdlib++-isystem /tmp/bar -stdlib=libc++ \
// RUN:   -fsyntax-only %s -### 2>&1 | FileCheck -check-prefix=INCPATH %s
// INCPATH: "-internal-isystem" "/tmp/foo" "-internal-isystem" "/tmp/bar"

// We shouldn't pass the -stdlib++-isystem to cc1.
// RUN: %clang -target aarch64-linux-gnu -ccc-install-dir %t/bin \
// RUN:   -stdlib++-isystem /tmp -stdlib=libc++ -fsyntax-only %s -### 2>&1 | \
// RUN:   FileCheck -check-prefix=NOCC1 %s
// RUN: %clang -target x86_64-apple-darwin -ccc-install-dir %t/bin \
// RUN:   -stdlib++-isystem /tmp -stdlib=libc++ -fsyntax-only %s -### 2>&1 | \
// RUN:   FileCheck -check-prefix=NOCC1 %s
// NOCC1-NOT: "-stdlib++-isystem" "/tmp"

// It should respect -nostdinc++.
// RUN: %clang -target aarch64-linux-gnu -ccc-install-dir %t/bin \
// RUN:   -stdlib++-isystem /tmp/foo -stdlib++-isystem /tmp/bar -nostdinc++ \
// RUN:   -fsyntax-only %s -### 2>&1 | FileCheck -check-prefix=NOSTDINCXX %s
// RUN: %clang -target x86_64-apple-darwin -ccc-install-dir %t/bin \
// RUN:   -stdlib++-isystem /tmp/foo -stdlib++-isystem /tmp/bar -nostdinc++ \
// RUN:   -fsyntax-only %s -### 2>&1 | FileCheck -check-prefix=NOSTDINCXX %s
// NOSTDINCXX-NOT: "-internal-isystem" "/tmp/foo" "-internal-isystem" "/tmp/bar"

// It should take effect even if -nostdinc or -nostdlibinc are specified; only
// -nostdinc++ should suppress it.
// RUN: %clang -target aarch64-linux-gnu -ccc-install-dir %t/bin \
// RUN:   -stdlib++-isystem /tmp/foo -stdlib++-isystem /tmp/bar -nostdinc \
// RUN:   -fsyntax-only %s -### 2>&1 | FileCheck -check-prefix=NOSTDINC %s
// RUN: %clang -target x86_64-apple-darwin -ccc-install-dir %t/bin \
// RUN:   -stdlib++-isystem /tmp/foo -stdlib++-isystem /tmp/bar -nostdlibinc \
// RUN:   -fsyntax-only %s -### 2>&1 | FileCheck -check-prefix=NOSTDINC %s
// NOSTDINC: "-internal-isystem" "/tmp/foo" "-internal-isystem" "/tmp/bar"
