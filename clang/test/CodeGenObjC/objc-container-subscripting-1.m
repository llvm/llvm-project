// RUN: %clang_cc1 -emit-llvm -triple x86_64-apple-darwin -o - %s | FileCheck %s

typedef unsigned int size_t;
@protocol P @end

@interface NSMutableArray
- (id)objectAtIndexedSubscript:(size_t)index;
- (void)setObject:(id)object atIndexedSubscript:(size_t)index;
@end

@interface NSMutableDictionary
- (id)objectForKeyedSubscript:(id)key;
- (void)setObject:(id)object forKeyedSubscript:(id)key;
@end

int main(void) {
  NSMutableArray *array;
  id val;

  id oldObject = array[10];
// CHECK: [[ARR:%.*]] = load {{.*}} [[array:%.*]], align 8
// CHECK-NEXT: [[SEL:%.*]] = load ptr, ptr @OBJC_SELECTOR_REFERENCES_
// CHECK-NEXT: [[CALL:%.*]] = call ptr @objc_msgSend(ptr noundef [[ARR]], ptr noundef [[SEL]], i32 noundef 10)
// CHECK-NEXT: store ptr [[CALL]], ptr [[OLDOBJ:%.*]], align 8

  val = (array[10] = oldObject);
// CHECK:      [[FOUR:%.*]] = load ptr, ptr [[oldObject:%.*]], align 8
// CHECK-NEXT: [[THREE:%.*]] = load {{.*}} [[array:%.*]], align 8
// CHECK-NEXT: [[FIVE:%.*]] = load ptr, ptr @OBJC_SELECTOR_REFERENCES_.2
// CHECK-NEXT: call void @objc_msgSend(ptr noundef [[THREE]], ptr noundef [[FIVE]], ptr noundef [[FOUR]], i32 noundef 10)
// CHECK-NEXT: store ptr [[FOUR]], ptr [[val:%.*]]

  NSMutableDictionary *dictionary;
  id key;
  id newObject;
  oldObject = dictionary[key];
// CHECK:  [[SEVEN:%.*]] = load {{.*}} [[DICTIONARY:%.*]], align 8
// CHECK-NEXT:  [[EIGHT:%.*]] = load ptr, ptr [[KEY:%.*]], align 8
// CHECK-NEXT:  [[TEN:%.*]] = load ptr, ptr @OBJC_SELECTOR_REFERENCES_.4
// CHECK-NEXT:  [[CALL1:%.*]] = call ptr @objc_msgSend(ptr noundef [[SEVEN]], ptr noundef [[TEN]], ptr noundef [[EIGHT]])
// CHECK-NEXT:  store ptr [[CALL1]], ptr [[oldObject:%.*]], align 8


  val = (dictionary[key] = newObject);
// CHECK:       [[FOURTEEN:%.*]] = load ptr, ptr [[NEWOBJECT:%.*]], align 8
// CHECK-NEXT:  [[TWELVE:%.*]] = load {{.*}} [[DICTIONARY]], align 8
// CHECK-NEXT:  [[THIRTEEN:%.*]] = load ptr, ptr [[KEY]], align 8
// CHECK-NEXT:  [[SIXTEEN:%.*]] = load ptr, ptr @OBJC_SELECTOR_REFERENCES_.6
// CHECK-NEXT:  call void @objc_msgSend(ptr noundef [[TWELVE]], ptr noundef [[SIXTEEN]], ptr noundef [[FOURTEEN]], ptr noundef [[THIRTEEN]])
// CHECK-NEXT: store ptr [[FOURTEEN]], ptr [[val:%.*]]
}

