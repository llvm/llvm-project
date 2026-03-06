// RUN: %clang -target x86_64-apple-ios13.1-macabi -isysroot %S/Inputs/MacOSX10.15.sdk -c %s -### 2>&1 \
// RUN:   | FileCheck %s

// CHECK: -fdepfile-entry={{.*}}MacOSX10.15.sdk{{/|\\\\}}SDKSettings.json
