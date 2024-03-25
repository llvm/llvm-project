// RUN: rm -rf %t.tmpdir
// RUN: mkdir -p %t.tmpdir/Xcode.app/Contents/Developers/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.14.sdk
// RUN: not %clang -### -target x86_64-apple-macos10.10 -fobjc-link-runtime -lfoo \
// RUN:   -isysroot %t.tmpdir/Xcode.app/Contents/Developers/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.14.sdk \
// RUN:   %s 2>&1 | FileCheck %s

// CHECK: error: SDK does not contain 'libarclite' at the path '
// CHECK: .tmpdir/Xcode.app/{{.*}}libarclite_macosx.a';
// CHECK: try increasing the minimum deployment target
// CHECK: -lfoo
// CHECK: .tmpdir/Xcode.app/{{.*}}libarclite_macosx.a
