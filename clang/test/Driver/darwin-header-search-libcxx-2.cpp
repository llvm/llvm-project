// General tests that the header search paths for libc++ detected by the driver
// and passed to CC1 are correct on Darwin platforms. This test copies the clang
// binary, which won't work if it uses any dynamic libraries (BUILD_SHARED_LIBS,
// LLVM_LINK_LLVM_DYLIB, or CLANG_LINK_CLANG_DYLIB).

// UNSUPPORTED: system-windows
// REQUIRES: static-libs

// ----------------------------------------------------------------------------
// On Darwin, libc++ can be installed in one of the following places:
// 1. Alongside the compiler in <install>/include/c++/v1
// 2. Alongside the compiler in <clang-executable-folder>/../include/c++/v1
// 3. In a SDK (or a custom sysroot) in <sysroot>/usr/include/c++/v1

// The build folders do not have an `include/c++/v1`; create a new
// local folder hierarchy that meets this requirement.
// Note: this might not work with weird RPATH configurations.
// RUN: rm -rf %t/install
// RUN: mkdir -p %t/install/bin
// RUN: cp %clang %t/install/bin/clang
// RUN: mkdir -p %t/install/include/c++/v1

// Headers in (1) and in (2) -> (1) is preferred over (2)
// RUN: rm -rf %t/symlinked1
// RUN: mkdir -p %t/symlinked1/bin
// RUN: ln -sf %t/install/bin/clang %t/symlinked1/bin/clang
// RUN: mkdir -p %t/symlinked1/include/c++/v1

// RUN: %t/symlinked1/bin/clang -### %s -no-canonical-prefixes -fsyntax-only 2>&1 \
// RUN:     --target=x86_64-apple-darwin \
// RUN:     -stdlib=libc++ \
// RUN:     -isysroot %S/Inputs/basic_darwin_sdk_usr_cxx_v1 \
// RUN:   | FileCheck -DSYMLINKED=%t/symlinked1 \
// RUN:               -DTOOLCHAIN=%t/install \
// RUN:               -DSYSROOT=%S/Inputs/basic_darwin_sdk_usr_cxx_v1 \
// RUN:               --check-prefix=CHECK-SYMLINKED-INCLUDE-CXX-V1 %s
// CHECK-SYMLINKED-INCLUDE-CXX-V1: "-internal-isystem" "[[SYMLINKED]]/bin/../include/c++/v1"
// CHECK-SYMLINKED-INCLUDE-CXX-V1-NOT: "-internal-isystem" "[[TOOLCHAIN]]/bin/../include/c++/v1"
// CHECK-SYMLINKED-INCLUDE-CXX-V1-NOT: "-internal-isystem" "[[SYSROOT]]/usr/include/c++/v1"

// Headers in (2) and in (3) -> (2) is preferred over (3)
// RUN: rm -rf %t/symlinked2
// RUN: mkdir -p %t/symlinked2/bin
// RUN: ln -sf %t/install/bin/clang %t/symlinked2/bin/clang

// RUN: %t/symlinked2/bin/clang -### %s -fsyntax-only 2>&1 \
// RUN:     --target=x86_64-apple-darwin \
// RUN:     -stdlib=libc++ \
// RUN:     -isysroot %S/Inputs/basic_darwin_sdk_usr_cxx_v1 \
// RUN:   | FileCheck -DTOOLCHAIN=%t/install \
// RUN:               -DSYSROOT=%S/Inputs/basic_darwin_sdk_usr_cxx_v1 \
// RUN:               --check-prefix=CHECK-TOOLCHAIN-INCLUDE-CXX-V1 %s
// CHECK-TOOLCHAIN-INCLUDE-CXX-V1: "-internal-isystem" "[[TOOLCHAIN]]/bin/../include/c++/v1"
// CHECK-TOOLCHAIN-INCLUDE-CXX-V1-NOT: "-internal-isystem" "[[SYSROOT]]/usr/include/c++/v1"

// Headers in (2) and nowhere else -> (2) is used
// RUN: %t/symlinked2/bin/clang -### %s -fsyntax-only 2>&1 \
// RUN:     --target=x86_64-apple-darwin \
// RUN:     -stdlib=libc++ \
// RUN:     -isysroot %S/Inputs/basic_darwin_sdk_usr_cxx_v1 \
// RUN:   | FileCheck -DTOOLCHAIN=%t/install \
// RUN:               -DSYSROOT=%S/Inputs/basic_darwin_sdk_no_libcxx \
// RUN:               --check-prefix=CHECK-TOOLCHAIN-NO-SYSROOT %s
// CHECK-TOOLCHAIN-NO-SYSROOT: "-internal-isystem" "[[TOOLCHAIN]]/bin/../include/c++/v1"
