// UNSUPPORTED: system-windows

// General tests that the header search paths for libc++ detected by the driver
// and passed to CC1 are correct on Darwin platforms.

// Check without a sysroot and without headers alongside the installation
// (no include path should be added, and no warning or error).
//
// RUN: %clang -no-canonical-prefixes %s -### -fsyntax-only 2>&1 \
// RUN:     -target x86_64-apple-darwin \
// RUN:     -stdlib=libc++ \
// RUN:     -ccc-install-dir %S/Inputs/basic_darwin_toolchain_no_libcxx/usr/bin \
// RUN:   | FileCheck --check-prefix=CHECK-LIBCXX-NONE %s
// CHECK-LIBCXX-NONE: "{{[^"]*}}clang{{[^"]*}}" "-cc1"

// Check with only headers alongside the installation (those should be used,
// but we should still add /usr/include/c++/v1 after to preserve legacy).
//
// RUN: %clang -no-canonical-prefixes %s -### -fsyntax-only 2>&1 \
// RUN:     -target x86_64-apple-darwin \
// RUN:     -stdlib=libc++ \
// RUN:     -ccc-install-dir %S/Inputs/basic_darwin_toolchain/usr/bin \
// RUN:   | FileCheck -DTOOLCHAIN=%S/Inputs/basic_darwin_toolchain --check-prefix=CHECK-LIBCXX-TOOLCHAIN-1 %s
// CHECK-LIBCXX-TOOLCHAIN-1: "{{[^"]*}}clang{{[^"]*}}" "-cc1"
// CHECK-LIBCXX-TOOLCHAIN-1: "-internal-isystem" "[[TOOLCHAIN]]/usr/bin/../include/c++/v1"
// CHECK-LIBCXX-TOOLCHAIN-1: "-internal-isystem" "/usr/include/c++/v1"
//
// RUN: %clang -no-canonical-prefixes %s -### -fsyntax-only 2>&1 \
// RUN:     -target x86_64-apple-darwin \
// RUN:     -stdlib=libc++ \
// RUN:     -ccc-install-dir %S/Inputs/basic_darwin_toolchain/usr/bin \
// RUN:     -isysroot %S/Inputs/basic_darwin_sdk_no_libcxx \
// RUN:   | FileCheck -DTOOLCHAIN=%S/Inputs/basic_darwin_toolchain --check-prefix=CHECK-LIBCXX-TOOLCHAIN-2 %s
// CHECK-LIBCXX-TOOLCHAIN-2: "{{[^"]*}}clang{{[^"]*}}" "-cc1"
// CHECK-LIBCXX-TOOLCHAIN-2: "-internal-isystem" "[[TOOLCHAIN]]/usr/bin/../include/c++/v1"

// Check with both headers in the sysroot and headers alongside the installation
// (the headers in <sysroot> should be added after the toolchain headers).
// Ensure that both -isysroot and --sysroot work, and that isysroot has precedence.
//
// RUN: %clang -no-canonical-prefixes %s -### -fsyntax-only 2>&1 \
// RUN:     -target x86_64-apple-darwin \
// RUN:     -stdlib=libc++ \
// RUN:     -ccc-install-dir %S/Inputs/basic_darwin_toolchain/usr/bin \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     -isysroot %S/Inputs/basic_darwin_sdk_usr \
// RUN:   | FileCheck -DSYSROOT=%S/Inputs/basic_darwin_sdk_usr \
// RUN:               -DTOOLCHAIN=%S/Inputs/basic_darwin_toolchain \
// RUN:               --check-prefix=CHECK-LIBCXX-SYSROOT_AND_TOOLCHAIN-1 %s
//
// RUN: %clang -no-canonical-prefixes %s -### -fsyntax-only 2>&1 \
// RUN:     -target x86_64-apple-darwin \
// RUN:     -stdlib=libc++ \
// RUN:     -ccc-install-dir %S/Inputs/basic_darwin_toolchain/usr/bin \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     --sysroot %S/Inputs/basic_darwin_sdk_usr \
// RUN:   | FileCheck -DSYSROOT=%S/Inputs/basic_darwin_sdk_usr \
// RUN:               -DTOOLCHAIN=%S/Inputs/basic_darwin_toolchain \
// RUN:               --check-prefix=CHECK-LIBCXX-SYSROOT_AND_TOOLCHAIN-1 %s
//
// RUN: %clang -no-canonical-prefixes %s -### -fsyntax-only 2>&1 \
// RUN:     -target x86_64-apple-darwin \
// RUN:     -stdlib=libc++ \
// RUN:     -ccc-install-dir %S/Inputs/basic_darwin_toolchain/usr/bin \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     -isysroot %S/Inputs/basic_darwin_sdk_usr \
// RUN:     --sysroot %S/Inputs/basic_darwin_sdk_no_libcxx \
// RUN:   | FileCheck -DSYSROOT=%S/Inputs/basic_darwin_sdk_usr \
// RUN:               -DTOOLCHAIN=%S/Inputs/basic_darwin_toolchain \
// RUN:               --check-prefix=CHECK-LIBCXX-SYSROOT_AND_TOOLCHAIN-1 %s
//
// CHECK-LIBCXX-SYSROOT_AND_TOOLCHAIN-1: "{{[^"]*}}clang{{[^"]*}}" "-cc1"
// CHECK-LIBCXX-SYSROOT_AND_TOOLCHAIN-1: "-internal-isystem" "[[TOOLCHAIN]]/usr/bin/../include/c++/v1"
// CHECK-LIBCXX-SYSROOT_AND_TOOLCHAIN-1: "-internal-isystem" "[[SYSROOT]]/usr/include/c++/v1"

// Make sure that using -nostdinc or -nostdlibinc will drop the non-toolchain
// C++ library include paths (so all except <toolchain>/usr/bin/../include/c++/v1).
//
// RUN: %clang -no-canonical-prefixes %s -### -fsyntax-only 2>&1 \
// RUN:     -target x86_64-apple-darwin16 \
// RUN:     -ccc-install-dir %S/Inputs/basic_darwin_toolchain/usr/bin \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     -isysroot %S/Inputs/basic_darwin_sdk_usr \
// RUN:     -stdlib=platform \
// RUN:     -nostdinc \
// RUN:   | FileCheck -DSYSROOT=%S/Inputs/basic_darwin_sdk_usr \
// RUN:               -DTOOLCHAIN=%S/Inputs/basic_darwin_toolchain \
// RUN:               --check-prefix=CHECK-LIBCXX-NOSTDINC %s
// CHECK-LIBCXX-NOSTDINC: "{{[^"]*}}clang{{[^"]*}}" "-cc1"
// CHECK-LIBCXX-NOSTDINC: "-internal-isystem" "[[TOOLCHAIN]]/usr/bin/../include/c++/v1"
// CHECK-LIBCXX-NOSTDINC-NOT: "-internal-isystem" "[[SYSROOT]]/usr/include/c++/v1"
//
// RUN: %clang -no-canonical-prefixes %s -### -fsyntax-only 2>&1 \
// RUN:     -target x86_64-apple-darwin16 \
// RUN:     -ccc-install-dir %S/Inputs/basic_darwin_toolchain/usr/bin \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     -isysroot %S/Inputs/basic_darwin_sdk_usr \
// RUN:     -stdlib=platform \
// RUN:     -nostdinc \
// RUN:   | FileCheck -DSYSROOT=%S/Inputs/basic_darwin_sdk_usr \
// RUN:               -DTOOLCHAIN=%S/Inputs/basic_darwin_toolchain \
// RUN:               --check-prefix=CHECK-LIBCXX-NOSTDLIBINC %s
// CHECK-LIBCXX-NOSTDLIBINC: "{{[^"]*}}clang{{[^"]*}}" "-cc1"
// CHECK-LIBCXX-NOSTDLIBINC: "-internal-isystem" "[[TOOLCHAIN]]/usr/bin/../include/c++/v1"
// CHECK-LIBCXX-NOSTDLIBINC-NOT: "-internal-isystem" "[[SYSROOT]]/usr/include/c++/v1"
