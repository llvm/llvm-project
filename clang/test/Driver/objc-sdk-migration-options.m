// Check miscellaneous Objective-C sdk migration options.
// rdar://19994452

// RUN: %clang  -objcmt-migrate-property-dot-syntax -target x86_64-apple-darwin10 -S -### %s \
// RUN:   -arch x86_64 2> %t
// RUN: FileCheck < %t %s

// CHECK: "-cc1"
// CHECK: -objcmt-migrate-property-dot-syntax
