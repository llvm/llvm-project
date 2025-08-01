// RUN: %clang_analyze_cc1 %s -analyzer-checker=core -analyzer-output=plist -analyzer-config serialize-stats=true -o %t.plist
// REQUIRES: asserts
// RUN: FileCheck --input-file=%t.plist %s

void foo(void) {}


// CHECK:  <key>diagnostics</key>
// CHECK-NEXT:  <array>
// CHECK-NEXT:  </array>
// CHECK-NEXT: <key>files</key>
// CHECK-NEXT: <array>
// CHECK-NEXT: </array>
// CHECK-NEXT: <key>statistics</key>
// CHECK-NEXT: <string>{
// CHECK: }
// CHECK-NEXT: </string>
