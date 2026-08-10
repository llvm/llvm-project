
@interface I2
@end

// RUN: env CINDEXTEST_EDITING=1 \
// RUN:   c-index-test -test-load-source-reparse 1 local %s | FileCheck %s

// CHECK: preamble-reparse-with-BOM.m:2:12: ObjCInterfaceDecl=I2:2:12 Extent=[2:1 - 3:5]
