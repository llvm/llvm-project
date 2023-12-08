// RUN: %clang_cc1 -fobjc-arc -emit-llvm -triple x86_64-apple-darwin -o - %s | FileCheck %s

@interface NSMutableArray
- (id)objectAtIndexedSubscript:(int)index;
- (void)setObject:(id)object atIndexedSubscript:(int)index;
@end

id func(void) {
  NSMutableArray *array;
  array[3] = 0;
  return array[3];
}

// CHECK: [[call:%.*]] = call ptr @objc_msgSend
// CHECK: [[SIX:%.*]] = notail call ptr @llvm.objc.retainAutoreleasedReturnValue(ptr [[call]]) [[NUW:#[0-9]+]]
// CHECK: call void @llvm.objc.storeStrong(ptr {{%.*}}, ptr null)
// CHECK: [[EIGHT:%.*]] = tail call ptr @llvm.objc.autoreleaseReturnValue(ptr [[SIX]]) [[NUW]]
// CHECK: ret ptr [[EIGHT]]

// CHECK: attributes [[NUW]] = { nounwind }
