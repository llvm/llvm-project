// UNSUPPORTED: system-windows

// General tests that the system header search paths detected by the driver
// and passed to CC1 are correct on Darwin platforms.

// Check system headers (everything below <sysroot> and <resource-dir>).  Ensure
// that both sysroot and isysroot are checked, and that isysroot has precedence.

// RUN: %clang -### %s -fsyntax-only 2>&1 \
// RUN:     --target=x86_64-apple-darwin \
// RUN:     -ccc-install-dir %S/Inputs/basic_darwin_toolchain_no_libcxx/usr/bin \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     -isysroot %S/Inputs/basic_darwin_sdk_usr_and_usr_local \
// RUN:   | FileCheck -DSYSROOT=%S/Inputs/basic_darwin_sdk_usr_and_usr_local \
// RUN:               -DRESOURCE=%S/Inputs/resource_dir \
// RUN:               --check-prefix=CHECK-SYSTEM %s

// RUN: %clang -### %s -fsyntax-only 2>&1 \
// RUN:     --target=x86_64-apple-darwin \
// RUN:     -ccc-install-dir %S/Inputs/basic_darwin_toolchain_no_libcxx/usr/bin \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     --sysroot %S/Inputs/basic_darwin_sdk_usr_and_usr_local \
// RUN:   | FileCheck -DSYSROOT=%S/Inputs/basic_darwin_sdk_usr_and_usr_local \
// RUN:               -DRESOURCE=%S/Inputs/resource_dir \
// RUN:               --check-prefix=CHECK-SYSTEM %s

// RUN: %clang -### %s -fsyntax-only 2>&1 \
// RUN:     --target=x86_64-apple-darwin \
// RUN:     -ccc-install-dir %S/Inputs/basic_darwin_toolchain_no_libcxx/usr/bin \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     -isysroot %S/Inputs/basic_darwin_sdk_usr_and_usr_local \
// RUN:     --sysroot / \
// RUN:   | FileCheck -DSYSROOT=%S/Inputs/basic_darwin_sdk_usr_and_usr_local \
// RUN:               -DRESOURCE=%S/Inputs/resource_dir \
// RUN:               --check-prefix=CHECK-SYSTEM %s

// CHECK-SYSTEM: "-cc1"
// CHECK-SYSTEM: "-internal-isystem" "[[SYSROOT]]/usr/local/include"
// CHECK-SYSTEM: "-internal-isystem" "[[RESOURCE]]/include"
// CHECK-SYSTEM: "-internal-externc-isystem" "[[SYSROOT]]/usr/include"

// Make sure that using -nobuiltininc will drop resource headers

// RUN: %clang -### %s -fsyntax-only 2>&1 \
// RUN:     --target=x86_64-apple-darwin \
// RUN:     -ccc-install-dir %S/Inputs/basic_darwin_toolchain_no_libcxx/usr/bin \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     -isysroot %S/Inputs/basic_darwin_sdk_usr_and_usr_local \
// RUN:     -nobuiltininc \
// RUN:   | FileCheck -DSYSROOT=%S/Inputs/basic_darwin_sdk_usr_and_usr_local \
// RUN:               -DRESOURCE=%S/Inputs/resource_dir \
// RUN:               --check-prefix=CHECK-NOBUILTININC %s
// CHECK-NOBUILTININC: "-cc1"
// CHECK-NOBUILTININC: "-internal-isystem" "[[SYSROOT]]/usr/local/include"
// CHECK-NOBUILTININC-NOT: "-internal-isystem" "[[RESOURCE]]/include"
// CHECK-NOBUILTININC: "-internal-externc-isystem" "[[SYSROOT]]/usr/include"

// Make sure that using -nostdlibinc will drop <sysroot>/usr/local/include and
// <sysroot>/usr/include.

// RUN: %clang -### %s -fsyntax-only 2>&1 \
// RUN:     --target=x86_64-apple-darwin \
// RUN:     -ccc-install-dir %S/Inputs/basic_darwin_toolchain_no_libcxx/usr/bin \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     -isysroot %S/Inputs/basic_darwin_sdk_usr_and_usr_local \
// RUN:     -nostdlibinc \
// RUN:   | FileCheck -DSYSROOT=%S/Inputs/basic_darwin_sdk_usr_and_usr_local \
// RUN:               -DRESOURCE=%S/Inputs/resource_dir \
// RUN:               --check-prefix=CHECK-NOSTDLIBINC %s
// CHECK-NOSTDLIBINC: "-cc1"
// CHECK-NOSTDLIBINC-NOT: "-internal-isystem" "[[SYSROOT]]/usr/local/include"
// CHECK-NOSTDLIBINC: "-internal-isystem" "[[RESOURCE]]/include"
// CHECK-NOSTDLIBINC-NOT: "-internal-externc-isystem" "[[SYSROOT]]/usr/include"

// Make sure that -nostdinc drops all the system include paths, including
// <resource>/include.

// RUN: %clang -### %s -fsyntax-only 2>&1 \
// RUN:     --target=x86_64-apple-darwin \
// RUN:     -ccc-install-dir %S/Inputs/basic_darwin_toolchain_no_libcxx/usr/bin \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     -isysroot %S/Inputs/basic_darwin_sdk_usr_and_usr_local \
// RUN:     -nostdinc \
// RUN:   | FileCheck -DSYSROOT=%S/Inputs/basic_darwin_sdk_usr_and_usr_local \
// RUN:               -DRESOURCE=%S/Inputs/resource_dir \
// RUN:               --check-prefix=CHECK-NOSTDINC %s
// CHECK-NOSTDINC: "-cc1"
// CHECK-NOSTDINC-NOT: "-internal-isystem" "[[SYSROOT]]/usr/local/include"
// CHECK-NOSTDINC-NOT: "-internal-isystem" "[[RESOURCE]]/include"
// CHECK-NOSTDINC-NOT: "-internal-externc-isystem" "[[SYSROOT]]/usr/include"

// Check search paths without -isysroot

// RUN: %clang -### %s -fsyntax-only 2>&1 \
// RUN:     --target=x86_64-apple-darwin \
// RUN:     -ccc-install-dir %S/Inputs/basic_darwin_toolchain_no_libcxx/usr/bin \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     --sysroot="" \
// RUN:   | FileCheck -DRESOURCE=%S/Inputs/resource_dir \
// RUN:               --check-prefix=CHECK-NOSYSROOT %s
// CHECK-NOSYSROOT: "-cc1"
// CHECK-NOSYSROOT: "-internal-isystem" "/usr/local/include"
// CHECK-NOSYSROOT: "-internal-isystem" "[[RESOURCE]]/include"
// CHECK-NOSYSROOT: "-internal-externc-isystem" "/usr/include"

// RUN: %clang -### %s -fsyntax-only 2>&1 \
// RUN:     --target=x86_64-apple-darwin \
// RUN:     -ccc-install-dir %S/Inputs/basic_darwin_toolchain_no_libcxx/usr/bin \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     -isysroot %S/Inputs/basic_darwin_sdk_usr_and_usr_local \
// RUN:     -nostdinc -ibuiltininc \
// RUN:   | FileCheck -DSYSROOT=%S/Inputs/basic_darwin_sdk_usr_and_usr_local \
// RUN:               -DRESOURCE=%S/Inputs/resource_dir \
// RUN:               --check-prefix=CHECK-NOSTDINC-BUILTINC %s
// RUN: %clang -### %s -fsyntax-only 2>&1 \
// RUN:     --target=x86_64-apple-darwin \
// RUN:     -ccc-install-dir %S/Inputs/basic_darwin_toolchain_no_libcxx/usr/bin \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     -isysroot %S/Inputs/basic_darwin_sdk_usr_and_usr_local \
// RUN:     -ibuiltininc -nostdinc \
// RUN:   | FileCheck -DSYSROOT=%S/Inputs/basic_darwin_sdk_usr_and_usr_local \
// RUN:               -DRESOURCE=%S/Inputs/resource_dir \
// RUN:               --check-prefix=CHECK-NOSTDINC-BUILTINC %s
// RUN: %clang -### %s -fsyntax-only 2>&1 \
// RUN:     --target=x86_64-apple-darwin \
// RUN:     -ccc-install-dir %S/Inputs/basic_darwin_toolchain_no_libcxx/usr/bin \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     -isysroot %S/Inputs/basic_darwin_sdk_usr_and_usr_local \
// RUN:     -nostdinc -nobuiltininc -ibuiltininc \
// RUN:   | FileCheck -DSYSROOT=%S/Inputs/basic_darwin_sdk_usr_and_usr_local \
// RUN:               -DRESOURCE=%S/Inputs/resource_dir \
// RUN:               --check-prefix=CHECK-NOSTDINC-BUILTINC %s
// CHECK-NOSTDINC-BUILTINC: "-cc1"
// CHECK-NOSTDINC-BUILTINC-NOT: "-internal-isystem" "[[SYSROOT]]/usr/local/include"
// CHECK-NOSTDINC-BUILTINC: "-internal-isystem" "[[RESOURCE]]/include"
// CHECK-NOSTDINC-BUILTINC-NOT: "-internal-externc-isystem" "[[SYSROOT]]/usr/include"

// RUN: %clang -### %s -fsyntax-only 2>&1 \
// RUN:     --target=x86_64-apple-darwin \
// RUN:     -ccc-install-dir %S/Inputs/basic_darwin_toolchain_no_libcxx/usr/bin \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     -isysroot %S/Inputs/basic_darwin_sdk_usr_and_usr_local \
// RUN:     -nobuiltininc -ibuiltininc \
// RUN:   | FileCheck -DSYSROOT=%S/Inputs/basic_darwin_sdk_usr_and_usr_local \
// RUN:               -DRESOURCE=%S/Inputs/resource_dir \
// RUN:               --check-prefix=CHECK-NOBUILTININC-BUILTINC %s
// CHECK-NOBUILTININC-BUILTINC: "-cc1"
// CHECK-NOBUILTININC-BUILTINC: "-internal-isystem" "[[SYSROOT]]/usr/local/include"
// CHECK-NOBUILTININC-BUILTINC: "-internal-isystem" "[[RESOURCE]]/include"
// CHECK-NOBUILTININC-BUILTINC: "-internal-externc-isystem" "[[SYSROOT]]/usr/include"

// RUN: %clang -### %s -fsyntax-only 2>&1 \
// RUN:     --target=x86_64-apple-darwin \
// RUN:     -ccc-install-dir %S/Inputs/basic_darwin_toolchain_no_libcxx/usr/bin \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     -isysroot %S/Inputs/basic_darwin_sdk_usr_and_usr_local \
// RUN:     -nostdinc -ibuiltininc -nobuiltininc \
// RUN:   | FileCheck -DSYSROOT=%S/Inputs/basic_darwin_sdk_usr_and_usr_local \
// RUN:               -DRESOURCE=%S/Inputs/resource_dir \
// RUN:               --check-prefix=CHECK-NOSTDINC-NO-BUILTINC %s
// CHECK-NOSTDINC-NO-BUILTINC: "-cc1"
// CHECK-NOSTDINC-NO-BUILTINC-NOT: "-internal-isystem" "[[SYSROOT]]/usr/local/include"
// CHECK-NOSTDINC-NO-BUILTINC-NOT: "-internal-isystem" "[[RESOURCE]]/include"
// CHECK-NOSTDINC-NO-BUILTINC-NOT: "-internal-externc-isystem" "[[SYSROOT]]/usr/include"

// RUN: %clang -### %s -fsyntax-only 2>&1 \
// RUN:     --target=x86_64-apple-darwin \
// RUN:     -ccc-install-dir %S/Inputs/basic_darwin_toolchain_no_libcxx/usr/bin \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     -isysroot %S/Inputs/basic_darwin_sdk_usr_and_usr_local \
// RUN:     -ibuiltininc -nobuiltininc \
// RUN:   | FileCheck -DSYSROOT=%S/Inputs/basic_darwin_sdk_usr_and_usr_local \
// RUN:               -DRESOURCE=%S/Inputs/resource_dir \
// RUN:               --check-prefix=CHECK-BUILTINC-NOBUILTININC %s
// CHECK-BUILTINC-NOBUILTININC: "-cc1"
// CHECK-BUILTINC-NOBUILTININC: "-internal-isystem" "[[SYSROOT]]/usr/local/include"
// CHECK-BUILTINC-NOBUILTININC-NOT: "-internal-isystem" "[[RESOURCE]]/include"
// CHECK-BUILTINC-NOBUILTININC: "-internal-externc-isystem" "[[SYSROOT]]/usr/include"
