// RUN: %clang_cc1 -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -triple x86_64-apple-darwin -fobjc-runtime=macosx-10.10.0 -fblocks -fobjc-arc %s | FileCheck %s

@interface Foo
@end
#define Bar Foo

@implementation Blah
// CHECK-LABEL: +[Blah load]:
+ load { // CHECK: File 0, [[@LINE]]:8 -> [[END:[0-9:]+]] = #0
  return 0;
  // CHECK: Expansion,File 0, [[@LINE+1]]:3 -> [[@LINE+1]]:6 = 0
  Bar *attribute; // CHECK: File 0, [[@LINE]]:6 -> [[END]] = 0
}
@end

int main() {}
