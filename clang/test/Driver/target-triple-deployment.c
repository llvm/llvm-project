// RUN: touch %t.o
// RUN: %clang -fuse-ld= -target x86_64-apple-macosx10.4 -mlinker-version=400 -### %t.o 2> %t.log
// RUN: %clang -fuse-ld= -target x86_64-apple-darwin9 -mlinker-version=400 -### %t.o 2>> %t.log
// RUN: %clang -fuse-ld= -target x86_64-apple-macosx10.7 -mlinker-version=400 -### %t.o 2>> %t.log

// RUN: %clang -fuse-ld= -target armv7-apple-ios -mlinker-version=400 -### %t.o 2>> %t.log
// RUN: %clang -fuse-ld= -target armv7-apple-ios0.0 -mlinker-version=400 -### %t.o 2>> %t.log
// RUN: %clang -fuse-ld= -target armv7-apple-ios1.2.3 -mlinker-version=400 -### %t.o 2>> %t.log
// RUN: %clang -fuse-ld= -target armv7-apple-ios5.0 -mlinker-version=400 -### %t.o 2>> %t.log
// RUN: %clang -fuse-ld= -target armv7-apple-ios7.0 -mlinker-version=400 -### %t.o 2>> %t.log
// RUN: %clang -fuse-ld= -target arm64-apple-ios -mlinker-version=400 -### %t.o 2>> %t.log
// RUN: %clang -fuse-ld= -target arm64e-apple-ios13.0 -mlinker-version=400 -### %t.o 2>> %t.log
// RUN: %clang -fuse-ld= -target arm64e-apple-ios14.1 -mlinker-version=400 -### %t.o 2>> %t.log

// RUN: FileCheck %s < %t.log

// CHECK: {{ld(.exe)?"}}
// CHECK: -macosx_version_min
// CHECK: 10.4.0
// CHECK: {{ld(.exe)?"}}
// CHECK: -macosx_version_min
// CHECK: 10.5.0
// CHECK: {{ld(.exe)?"}}
// CHECK: -macosx_version_min
// CHECK: 10.7.0
// CHECK: {{ld(.exe)?"}}
// CHECK: -iphoneos_version_min
// CHECK: 5.0.0
// CHECK: {{ld(.exe)?"}}
// CHECK: -iphoneos_version_min
// CHECK: 5.0.0
// CHECK: {{ld(.exe)?"}}
// CHECK: -iphoneos_version_min
// CHECK: 1.2.3
// CHECK: {{ld(.exe)?"}}
// CHECK: -iphoneos_version_min
// CHECK: 5.0.0
// CHECK: {{ld(.exe)?"}}
// CHECK: -iphoneos_version_min
// CHECK: 7.0.0
// CHECK: {{ld(.exe)?"}}
// CHECK: -iphoneos_version_min
// CHECK: 7.0.0
// CHECK: {{ld(.exe)?"}}
// CHECK: -iphoneos_version_min
// CHECK: 14.0.0
// CHECK: {{ld(.exe)?"}}
// CHECK: -iphoneos_version_min
// CHECK: 14.1.0
