// RUN: %clang --target=x86_64-apple-macosx10.7.0 -rewrite-objc %s -o - -### 2>&1 | \
// RUN:   FileCheck %s
//
// Check that we're running a preprocessing stage passing a not-preprocessed objective-c++ file as input
// CHECK: "-E"{{.*}}"-x" "objective-c++"
