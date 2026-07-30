// Verify that the analyzer gets the same flags as normal compilation
// (at least for a few key ones).

// RUN: env MACOSX_DEPLOYMENT_TARGET=10.5 %clang -target i386-apple-darwin9  -### --analyze -o /dev/null %s -msse 2> %t.log
// RUN: FileCheck --input-file=%t.log %s

// CHECK: "-analyze"
// CHECK: "-target-feature" "+sse"
