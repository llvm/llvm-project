// RUN: %clang_cc1 -arcmt-action=migrate -mt-migrate-directory %t -arcmt-migrate-emit-errors %s 2>&1 | FileCheck %s
// RUN: rm -rf %t

@protocol NSObject
- (oneway void)release;
@end

void test(id p) {
  [p release];
}

// CHECK: error: ARC forbids explicit message send of 'release'
