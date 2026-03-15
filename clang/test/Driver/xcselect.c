// REQUIRES: xcselect
// RUN: %clang -target arm64-apple-macosx -c -### %s 2> %t.log
// RUN: FileCheck %s <%t.log

// CHECK: "-isysroot" "{{.*}}/SDKs/MacOSX{{([0-9]+(\.[0-9]+)?)?}}.sdk"
