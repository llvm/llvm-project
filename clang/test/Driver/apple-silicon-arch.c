// RUN: env SDKROOT="/" %clang -arch arm64 -c -### %s 2>&1 | \
// RUN:   FileCheck %s
// RUN: env SDKROOT="/" %clang -arch arm64e -c -### %s 2>&1 | \
// RUN:   FileCheck --check-prefix=ARM64E %s
//
// REQUIRES: apple-silicon-mac
//
// CHECK: "-triple" "arm64-apple-macosx{{[0-9.]+}}"
// ARM64E: "-triple" "arm64e-apple-macosx{{[0-9.]+}}"
