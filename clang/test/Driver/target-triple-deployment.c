// RUN: touch %t.o
// RUN: %clang -target x86_64-apple-macosx -### %t.o 2> %t.log
// RUN: %clang -target x86_64-apple-darwin9 -### %t.o 2>> %t.log
// RUN: %clang -target x86_64-apple-macosx10.7 -### %t.o 2>> %t.log
//
// RUN: %clang -target armv7-apple-ios -### %t.o 2>> %t.log
// RUN: %clang -target armv7-apple-ios0.0 -### %t.o 2>> %t.log
// RUN: %clang -target armv7-apple-ios1.2.3 -### %t.o 2>> %t.log
// RUN: %clang -target armv7-apple-ios5.0 -### %t.o 2>> %t.log
// RUN: %clang -target armv7-apple-ios7.0 -### %t.o 2>> %t.log
// RUN: %clang -target arm64-apple-ios -### %t.o 2>> %t.log
//
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
