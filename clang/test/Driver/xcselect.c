// REQUIRES: xcselect
// RUN: %clang -target arm64-apple-darwin -c -### %s 2> %t.log
// RUN: FileCheck %s <%t.log

// CHECK: "-isysroot" "{{.*}}/SDKs/MacOSX{{([0-9]+(\.[0-9]+)?)?}}.sdk"
