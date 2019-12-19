// RUN: %clang -o /dev/null -v -fxray-instrument -target x86_64-apple-macos10.11 -c %s
// RUN: %clang -o /dev/null -v -fxray-instrument -target x86_64-apple-darwin15 -c %s
// REQUIRES: x86_64 || x86_64h
typedef int a;
