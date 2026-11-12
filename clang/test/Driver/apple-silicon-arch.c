// RUN: env SDKROOT="/" %clang -arch arm64 -c -### %s 2>&1 | \
// RUN:   FileCheck %s
//
// REQUIRES: apple-silicon-mac
//
// CHECK: "-triple" "arm64-apple-macosx{{[0-9.]+}}"
