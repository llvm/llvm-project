// REQUIRES: xcselect

// xcselect injects -isysroot for -macosx triples.
// RUN: %clang -target arm64-apple-macosx -c -### %s 2>&1 | \
// RUN:   FileCheck --check-prefix=SDK %s
// RUN: %clang -target arm64-apple-macosx15 -c -### %s 2>&1 | \
// RUN:   FileCheck --check-prefix=SDK %s
// RUN: %clang -target arm64-apple-macos -c -### %s 2>&1 | \
// RUN:   FileCheck --check-prefix=SDK %s
// RUN: %clang -target arm64-apple-macos26 -c -### %s 2>&1 | \
// RUN:   FileCheck --check-prefix=SDK %s
// RUN: %clang -target x86_64-apple-darwin -c -### %s 2>&1 | \
// RUN:   FileCheck --check-prefix=SDK %s
// RUN: %clang -target arm64-apple-darwin20 -c -### %s 2>&1 | \
// RUN:   FileCheck --check-prefix=SDK %s

// SDK: "-isysroot" "{{.*}}/SDKs/MacOSX{{([0-9]+(\.[0-9]+)?)?}}.sdk"

// RUN: %clang -target arm64-apple-macosx -c --no-xcselect -### %s 2>&1 | \
// RUN:   FileCheck --check-prefix=NO-SDK %s
// RUN: %clang -target armv7-apple-darwin10 -c -### %s 2>&1 | \
// RUN:   FileCheck --check-prefix=NO-SDK %s
// RUN: %clang -target arm64-apple-ios18 -c -### %s 2>&1 | \
// RUN:   FileCheck --check-prefix=NO-SDK %s
// RUN: %clang -target arm64-apple-darwin -mios-simulator-version-min=15.0 \
// RUN:   -c -### %s 2>&1 | \
// RUN:   FileCheck --check-prefix=NO-SDK %s
// RUN: env IPHONEOS_DEPLOYMENT_TARGET=14.0 \
// RUN:   %clang -target x86_64-apple-darwin -c -### %s 2>&1 | \
// RUN:   FileCheck --check-prefix=NO-SDK %s

// NO-SDK-NOT: "-isysroot"
