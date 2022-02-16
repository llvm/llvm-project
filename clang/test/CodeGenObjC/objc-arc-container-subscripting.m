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

// CHECK: [[call:%.*]] = call i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend
// CHECK: [[SIX:%.*]] = notail call i8* @llvm.objc.retainAutoreleasedReturnValue(i8* [[call]]) [[NUW:#[0-9]+]]
// CHECK: [[ARRAY_CASTED:%.*]] = bitcast %0** {{%.*}} to i8**
// CHECK: call void @llvm.objc.storeStrong(i8** [[ARRAY_CASTED]], i8* null)
// CHECK: [[EIGHT:%.*]] = tail call i8* @llvm.objc.autoreleaseReturnValue(i8* [[SIX]]) [[NUW]]
// CHECK: ret i8* [[EIGHT]]

// CHECK: attributes [[NUW]] = { nounwind }
