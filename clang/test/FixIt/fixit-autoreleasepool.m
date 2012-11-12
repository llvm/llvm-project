// RUN: %clang_cc1 -triple x86_64-apple-darwin10  -fdiagnostics-parseable-fixits -x objective-c %s 2>&1 | FileCheck %s
// rdar://10723084

void f0() {
  @autorelease {
  } 
}

// CHECK: {5:4-5:15}:"autoreleasepool"
