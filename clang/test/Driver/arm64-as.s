// Make sure the arm64 default on cyclone when compiling for apple.
// RUN: %clang -target arm64-apple-ios -arch arm64 -### -c %s 2>&1 | FileCheck -check-prefix=TARGET %s

// TARGET: "-cc1as"
// TARGET: "-target-cpu" "apple-a7"
