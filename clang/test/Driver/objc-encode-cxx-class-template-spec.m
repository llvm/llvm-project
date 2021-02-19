// RUN: %clang -target arm64-apple-ios11 -### %s -o - 2>&1 | FileCheck -check-prefix=DISABLE-ENC %s
// RUN: %clang -target arm64-apple-ios11 -fobjc-encode-cxx-class-template-spec -### %s -o - 2>&1 | FileCheck -check-prefix=ENABLE-ENC %s
// RUN: %clang -target x86_64-linux-gnu -fobjc-runtime=gnustep -### %s -o - 2>&1 | FileCheck -check-prefix=ENABLE-ENC %s
// RUN: %clang -target x86_64-linux-gnu -fobjc-runtime=gnustep -fno-objc-encode-cxx-class-template-spec -### %s -o - 2>&1 |  FileCheck -check-prefix=DISABLE-ENC %s

// DISABLE-ENC-NOT: -fobjc-encode-cxx-class-template-spec
// ENABLE-ENC: -fobjc-encode-cxx-class-template-spec
