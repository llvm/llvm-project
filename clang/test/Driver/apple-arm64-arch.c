// RUN: env SDKROOT="/" %clang -arch arm64 -c -### %s 2>&1 | \
// RUN:   FileCheck %s
//
// XFAIL: apple-silicon-mac
//
// CHECK: "-triple" "arm64-apple-ios{{[0-9.]+}}"
