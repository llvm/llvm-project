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

// CHECK: %call1 = call ptr @objc_msgSend(ptr noundef %2, ptr noundef %3, i32 noundef 3) [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
// CHECK: call void (...) @llvm.objc.clang.arc.noop.use(ptr %call1) #2
// CHECK: call void @llvm.objc.storeStrong(ptr {{%.*}}, ptr null)
// CHECK: %4 = tail call ptr @llvm.objc.autoreleaseReturnValue(ptr %call1) #2
// CHECK: ret ptr %4

// CHECK: attributes #2 = { nounwind }
