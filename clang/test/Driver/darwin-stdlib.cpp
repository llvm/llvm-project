// RUN: %clang -target x86_64-apple-darwin -arch arm64 -miphoneos-version-min=7.0 %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-LIBCXX
// RUN: %clang -target x86_64-apple-darwin -mmacosx-version-min=10.8 %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-LIBSTDCXX
// RUN: %clang -target x86_64-apple-darwin -mmacosx-version-min=10.9 %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-LIBCXX
// RUN: %clang -target x86_64-apple-darwin -arch armv7s -miphoneos-version-min=6.1 %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-LIBSTDCXX
// RUN: %clang -target x86_64-apple-darwin -arch armv7s -miphoneos-version-min=7.0 %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-LIBCXX
// RUN: %clang -target x86_64-apple-darwin -arch armv7k %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-LIBCXX

// The purpose of this test is that the libc++ headers should be found
// properly. At the moment this is done by passing -stdlib=libc++ down to the
// cc1 invocation. If and when we change to finding them in the driver this test
// should reflect that.

// CHECK-LIBCXX: -stdlib=libc++

// CHECK-LIBSTDCXX-NOT: -stdlib=libc++
// CHECK-LIBSTDCXX-NOT: -stdlib=libstdc++
