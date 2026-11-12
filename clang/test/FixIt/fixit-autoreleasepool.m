// RUN: not %clang_cc1 -triple x86_64-apple-darwin10  -fdiagnostics-parseable-fixits -x objective-c %s 2>&1 | FileCheck %s

void f0() {
  @autorelease {
  } 
}

// CHECK: {4:4-4:15}:"autoreleasepool"
