// UNSUPPORTED: system-windows

// General tests that the header search paths for libc++ detected by the driver
// and passed to CC1 are correct on Darwin platforms.

// Check without a sysroot and without headers alongside the installation
// (no include path should be added, and no warning or error).
//
// RUN: %clang -### %s -fsyntax-only 2>&1 \
// RUN:     --target=x86_64-apple-darwin \
// RUN:     -stdlib=libc++ \
// RUN:     -ccc-install-dir %S/Inputs/basic_darwin_toolchain_no_libcxx/usr/bin \
// RUN:   | FileCheck --check-prefix=CHECK-LIBCXX-NONE %s
// CHECK-LIBCXX-NONE: "-cc1"

// Check with only headers alongside the installation (those should be used).
//
// RUN: %clang -### %s -fsyntax-only 2>&1 \
// RUN:     --target=x86_64-apple-darwin \
// RUN:     -stdlib=libc++ \
// RUN:     -ccc-install-dir %S/Inputs/basic_darwin_toolchain/usr/bin \
// RUN:     --sysroot="" \
// RUN:   | FileCheck -DTOOLCHAIN=%S/Inputs/basic_darwin_toolchain \
// RUN:               --check-prefix=CHECK-LIBCXX-TOOLCHAIN-1 %s
// CHECK-LIBCXX-TOOLCHAIN-1: "-cc1"
// CHECK-LIBCXX-TOOLCHAIN-1: "-internal-isystem" "[[TOOLCHAIN]]/usr/bin/../include/c++/v1"
// CHECK-LIBCXX-TOOLCHAIN-1-NOT: "-internal-isystem" "/usr/include/c++/v1"
//
// RUN: %clang -### %s -fsyntax-only 2>&1 \
// RUN:     --target=x86_64-apple-darwin \
// RUN:     -stdlib=libc++ \
// RUN:     -ccc-install-dir %S/Inputs/basic_darwin_toolchain/usr/bin \
// RUN:     -isysroot %S/Inputs/basic_darwin_sdk_no_libcxx \
// RUN:   | FileCheck -DTOOLCHAIN=%S/Inputs/basic_darwin_toolchain \
// RUN:               -DSYSROOT=%S/Inputs/basic_darwin_sdk_no_libcxx \
// RUN:               --check-prefix=CHECK-LIBCXX-TOOLCHAIN-2 %s
// CHECK-LIBCXX-TOOLCHAIN-2: "-cc1"
// CHECK-LIBCXX-TOOLCHAIN-2: "-internal-isystem" "[[TOOLCHAIN]]/usr/bin/../include/c++/v1"
// CHECK-LIBCXX-TOOLCHAIN-2-NOT: "-internal-isystem" "[[SYSROOT]]/usr/include/c++/v1"

// Check with only headers in the sysroot (those should be used).
//
// RUN: %clang -### %s -fsyntax-only 2>&1 \
// RUN:     --target=x86_64-apple-darwin \
// RUN:     -stdlib=libc++ \
// RUN:     -ccc-install-dir %S/Inputs/basic_darwin_toolchain_no_libcxx/usr/bin \
// RUN:     -isysroot %S/Inputs/basic_darwin_sdk_usr_cxx_v1 \
// RUN:   | FileCheck -DSYSROOT=%S/Inputs/basic_darwin_sdk_usr_cxx_v1 \
// RUN:               -DTOOLCHAIN=%S/Inputs/basic_darwin_toolchain_no_libcxx \
// RUN:               --check-prefix=CHECK-LIBCXX-SYSROOT-1 %s
// CHECK-LIBCXX-SYSROOT-1: "-cc1"
// CHECK-LIBCXX-SYSROOT-1: "-internal-isystem" "[[SYSROOT]]/usr/include/c++/v1"
// CHECK-LIBCXX-SYSROOT-1-NOT: "-internal-isystem" "[[TOOLCHAIN]]/usr/bin/../include/c++/v1"

// Check with both headers in the sysroot and headers alongside the installation
// (the headers in the toolchain should be preferred over the <sysroot> headers).
// Ensure that both -isysroot and --sysroot work, and that isysroot has precedence
// over --sysroot.
//
// RUN: %clang -### %s -fsyntax-only 2>&1 \
// RUN:     --target=x86_64-apple-darwin \
// RUN:     -stdlib=libc++ \
// RUN:     -ccc-install-dir %S/Inputs/basic_darwin_toolchain/usr/bin \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     -isysroot %S/Inputs/basic_darwin_sdk_usr_cxx_v1 \
// RUN:   | FileCheck -DSYSROOT=%S/Inputs/basic_darwin_sdk_usr_cxx_v1 \
// RUN:               -DTOOLCHAIN=%S/Inputs/basic_darwin_toolchain \
// RUN:               --check-prefix=CHECK-LIBCXX-SYSROOT_AND_TOOLCHAIN-1 %s
//
// RUN: %clang -### %s -fsyntax-only 2>&1 \
// RUN:     --target=x86_64-apple-darwin \
// RUN:     -stdlib=libc++ \
// RUN:     -ccc-install-dir %S/Inputs/basic_darwin_toolchain/usr/bin \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     --sysroot %S/Inputs/basic_darwin_sdk_usr_cxx_v1 \
// RUN:   | FileCheck -DSYSROOT=%S/Inputs/basic_darwin_sdk_usr_cxx_v1 \
// RUN:               -DTOOLCHAIN=%S/Inputs/basic_darwin_toolchain \
// RUN:               --check-prefix=CHECK-LIBCXX-SYSROOT_AND_TOOLCHAIN-1 %s
//
// RUN: %clang -### %s -fsyntax-only 2>&1 \
// RUN:     --target=x86_64-apple-darwin \
// RUN:     -stdlib=libc++ \
// RUN:     -ccc-install-dir %S/Inputs/basic_darwin_toolchain/usr/bin \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     -isysroot %S/Inputs/basic_darwin_sdk_usr_cxx_v1 \
// RUN:     --sysroot %S/Inputs/basic_darwin_sdk_no_libcxx \
// RUN:   | FileCheck -DSYSROOT=%S/Inputs/basic_darwin_sdk_usr_cxx_v1 \
// RUN:               -DTOOLCHAIN=%S/Inputs/basic_darwin_toolchain \
// RUN:               --check-prefix=CHECK-LIBCXX-SYSROOT_AND_TOOLCHAIN-1 %s
//
// CHECK-LIBCXX-SYSROOT_AND_TOOLCHAIN-1: "-cc1"
// CHECK-LIBCXX-SYSROOT_AND_TOOLCHAIN-1: "-internal-isystem" "[[TOOLCHAIN]]/usr/bin/../include/c++/v1"
// CHECK-LIBCXX-SYSROOT_AND_TOOLCHAIN-1-NOT: "-internal-isystem" "[[SYSROOT]]/usr/include/c++/v1"

// Make sure that using -nostdinc, -nostdinc++ or -nostdlib will drop both the toolchain
// C++ include path and the sysroot one.
//
// RUN: %clang -### %s -fsyntax-only 2>&1 \
// RUN:     --target=x86_64-apple-darwin16 \
// RUN:     -ccc-install-dir %S/Inputs/basic_darwin_toolchain/usr/bin \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     -isysroot %S/Inputs/basic_darwin_sdk_usr \
// RUN:     -stdlib=platform \
// RUN:     -nostdinc \
// RUN:   | FileCheck -DSYSROOT=%S/Inputs/basic_darwin_sdk_usr \
// RUN:               -DTOOLCHAIN=%S/Inputs/basic_darwin_toolchain \
// RUN:               --check-prefix=CHECK-LIBCXX-NOSTDLIBINC %s
// CHECK-LIBCXX-NOSTDINC: "-cc1"
// CHECK-LIBCXX-NOSTDINC-NOT: "-internal-isystem" "[[TOOLCHAIN]]/usr/bin/../include/c++/v1"
// CHECK-LIBCXX-NOSTDINC-NOT: "-internal-isystem" "[[SYSROOT]]/usr/include/c++/v1"
//
// RUN: %clang -### %s -fsyntax-only 2>&1 \
// RUN:     --target=x86_64-apple-darwin16 \
// RUN:     -ccc-install-dir %S/Inputs/basic_darwin_toolchain/usr/bin \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     -isysroot %S/Inputs/basic_darwin_sdk_usr_cxx_v1 \
// RUN:     -stdlib=platform \
// RUN:     -nostdinc++ \
// RUN:   | FileCheck -DSYSROOT=%S/Inputs/basic_darwin_sdk_usr \
// RUN:               -DTOOLCHAIN=%S/Inputs/basic_darwin_toolchain \
// RUN:               --check-prefix=CHECK-LIBCXX-NOSTDINCXX %s
// CHECK-LIBCXX-NOSTDINCXX: "-cc1"
// CHECK-LIBCXX-NOSTDINCXX-NOT: "-internal-isystem" "[[TOOLCHAIN]]/usr/bin/../include/c++/v1"
// CHECK-LIBCXX-NOSTDINCXX-NOT: "-internal-isystem" "[[SYSROOT]]/usr/include/c++/v1"
//
// RUN: %clang -### %s -fsyntax-only 2>&1 \
// RUN:     --target=x86_64-apple-darwin16 \
// RUN:     -ccc-install-dir %S/Inputs/basic_darwin_toolchain/usr/bin \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     -isysroot %S/Inputs/basic_darwin_sdk_usr \
// RUN:     -stdlib=platform \
// RUN:     -nostdlibinc \
// RUN:   | FileCheck -DSYSROOT=%S/Inputs/basic_darwin_sdk_usr \
// RUN:               -DTOOLCHAIN=%S/Inputs/basic_darwin_toolchain \
// RUN:               --check-prefix=CHECK-LIBCXX-NOSTDLIBINC %s
// CHECK-LIBCXX-NOSTDLIBINC: "-cc1"
// CHECK-LIBCXX-NOSTDLIBINC-NOT: "-internal-isystem" "[[TOOLCHAIN]]/usr/bin/../include/c++/v1"
// CHECK-LIBCXX-NOSTDLIBINC-NOT: "-internal-isystem" "[[SYSROOT]]/usr/include/c++/v1"

// Make sure we explain that we considered a path but didn't add it when it
// doesn't exist.
//
// RUN: %clang %s -fsyntax-only -v 2>&1 \
// RUN:     --target=x86_64-apple-darwin \
// RUN:     -ccc-install-dir %S/Inputs/basic_darwin_toolchain_no_libcxx/usr/bin \
// RUN:     -isysroot %S/Inputs/basic_darwin_sdk \
// RUN:     -stdlib=libc++ \
// RUN:   | FileCheck -DTOOLCHAIN=%S/Inputs/basic_darwin_toolchain_no_libcxx \
// RUN:               --check-prefix=CHECK-LIBCXX-MISSING-TOOLCHAIN %s
// CHECK-LIBCXX-MISSING-TOOLCHAIN: ignoring nonexistent directory "[[TOOLCHAIN]]/usr/bin/../include/c++/v1"
//
// RUN: %clang %s -fsyntax-only -v 2>&1 \
// RUN:     --target=x86_64-apple-darwin \
// RUN:     -ccc-install-dir %S/Inputs/basic_darwin_toolchain_no_libcxx/usr/bin \
// RUN:     -isysroot %S/Inputs/basic_darwin_sdk_no_libcxx \
// RUN:     -stdlib=libc++ \
// RUN:   | FileCheck -DSYSROOT=%S/Inputs/basic_darwin_sdk_no_libcxx \
// RUN:               -DTOOLCHAIN=%S/Inputs/basic_darwin_toolchain_no_libcxx \
// RUN:               --check-prefix=CHECK-LIBCXX-MISSING-BOTH %s
// CHECK-LIBCXX-MISSING-BOTH: ignoring nonexistent directory "[[TOOLCHAIN]]/usr/bin/../include/c++/v1"
// CHECK-LIBCXX-MISSING-BOTH: ignoring nonexistent directory "[[SYSROOT]]/usr/include/c++/v1"

// Make sure that on Darwin, we use libc++ header search paths by default even when
// -stdlib= isn't passed.
//
// RUN: %clang -### %s -fsyntax-only 2>&1 \
// RUN:     --target=x86_64-apple-darwin \
// RUN:     -ccc-install-dir %S/Inputs/basic_darwin_toolchain_no_libcxx/usr/bin \
// RUN:     -isysroot %S/Inputs/basic_darwin_sdk_usr_cxx_v1 \
// RUN:   | FileCheck -DSYSROOT=%S/Inputs/basic_darwin_sdk_usr_cxx_v1 \
// RUN:               -DTOOLCHAIN=%S/Inputs/basic_darwin_toolchain_no_libcxx \
// RUN:               --check-prefix=CHECK-LIBCXX-STDLIB-UNSPECIFIED %s
// CHECK-LIBCXX-STDLIB-UNSPECIFIED: "-cc1"
// CHECK-LIBCXX-STDLIB-UNSPECIFIED: "-internal-isystem" "[[SYSROOT]]/usr/include/c++/v1"
