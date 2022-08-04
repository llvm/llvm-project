// This test will fail if CLANG_DEFAULT_CXX_STDLIB is set to anything different
// than the platform default. (see https://llvm.org/bugs/show_bug.cgi?id=30548)
// XFAIL: default-cxx-stdlib-set

// RUN: %clang -target x86_64-apple-darwin -ccc-install-dir %S/Inputs/darwin_toolchain_tree/bin/ -arch arm64 -miphoneos-version-min=7.0 %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-LIBCXX
// RUN: %clang -target x86_64-apple-darwin -ccc-install-dir %S/Inputs/darwin_toolchain_tree/bin/ -mmacosx-version-min=10.8 -Wno-stdlibcxx-not-found %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-LIBSTDCXX
// RUN: %clang -target x86_64-apple-darwin -ccc-install-dir %S/Inputs/darwin_toolchain_tree/bin/ -mmacosx-version-min=10.9 %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-LIBCXX
// RUN: %clang -target x86_64-apple-darwin -ccc-install-dir %S/Inputs/darwin_toolchain_tree/bin/ -arch armv7s -miphoneos-version-min=6.1 -Wno-stdlibcxx-not-found %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-LIBSTDCXX
// RUN: %clang -target x86_64-apple-darwin -ccc-install-dir %S/Inputs/darwin_toolchain_tree/bin/ -arch armv7s -miphoneos-version-min=7.0 %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-LIBCXX
// RUN: %clang -target x86_64-apple-darwin -ccc-install-dir %S/Inputs/darwin_toolchain_tree/bin/ -arch armv7k %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-LIBCXX

// CHECK-LIBCXX: "-stdlib=libc++"
// CHECK-LIBSTDCXX-NOT: -stdlib=libc++
// CHECK-LIBSTDCXX-NOT: -stdlib=libstdc++
