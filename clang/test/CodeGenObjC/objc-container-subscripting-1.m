// RUN: %clang_cc1 -no-opaque-pointers -emit-llvm -triple x86_64-apple-darwin -o - %s | FileCheck %s

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
// CHECK: [[ARR:%.*]] = load {{%.*}} [[array:%.*]], align 8
// CHECK-NEXT: [[ARRC:%.*]] = bitcast {{%.*}} [[ARR]] to i8*
// CHECK-NEXT: [[SEL:%.*]] = load i8*, i8** @OBJC_SELECTOR_REFERENCES_
// CHECK-NEXT: [[CALL:%.*]] = call i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8* (i8*, i8*, i32)*)(i8* noundef [[ARRC]], i8* noundef [[SEL]], i32 noundef 10)
// CHECK-NEXT: store i8* [[CALL]], i8** [[OLDOBJ:%.*]], align 8

  val = (array[10] = oldObject);
// CHECK:      [[FOUR:%.*]] = load i8*, i8** [[oldObject:%.*]], align 8
// CHECK-NEXT: [[THREE:%.*]] = load {{%.*}} [[array:%.*]], align 8
// CHECK-NEXT: [[SIX:%.*]] = bitcast {{%.*}} [[THREE]] to i8*
// CHECK-NEXT: [[FIVE:%.*]] = load i8*, i8** @OBJC_SELECTOR_REFERENCES_.2
// CHECK-NEXT: call void bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to void (i8*, i8*, i8*, i32)*)(i8* noundef [[SIX]], i8* noundef [[FIVE]], i8* noundef [[FOUR]], i32 noundef 10)
// CHECK-NEXT: store i8* [[FOUR]], i8** [[val:%.*]]

  NSMutableDictionary *dictionary;
  id key;
  id newObject;
  oldObject = dictionary[key];
// CHECK:  [[SEVEN:%.*]] = load {{%.*}} [[DICTIONARY:%.*]], align 8
// CHECK-NEXT:  [[EIGHT:%.*]] = load i8*, i8** [[KEY:%.*]], align 8
// CHECK-NEXT:  [[ELEVEN:%.*]] = bitcast {{%.*}} [[SEVEN]] to i8*
// CHECK-NEXT:  [[TEN:%.*]] = load i8*, i8** @OBJC_SELECTOR_REFERENCES_.4
// CHECK-NEXT:  [[CALL1:%.*]] = call i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8* (i8*, i8*, i8*)*)(i8* noundef [[ELEVEN]], i8* noundef [[TEN]], i8* noundef [[EIGHT]])
// CHECK-NEXT:  store i8* [[CALL1]], i8** [[oldObject:%.*]], align 8


  val = (dictionary[key] = newObject);
// CHECK:       [[FOURTEEN:%.*]] = load i8*, i8** [[NEWOBJECT:%.*]], align 8
// CHECK-NEXT:  [[TWELVE:%.*]] = load {{%.*}} [[DICTIONARY]], align 8
// CHECK-NEXT:  [[THIRTEEN:%.*]] = load i8*, i8** [[KEY]], align 8
// CHECK-NEXT:  [[SEVENTEEN:%.*]] = bitcast {{%.*}} [[TWELVE]] to i8*
// CHECK-NEXT:  [[SIXTEEN:%.*]] = load i8*, i8** @OBJC_SELECTOR_REFERENCES_.6
// CHECK-NEXT:  call void bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to void (i8*, i8*, i8*, i8*)*)(i8* noundef [[SEVENTEEN]], i8* noundef [[SIXTEEN]], i8* noundef [[FOURTEEN]], i8* noundef [[THIRTEEN]])
// CHECK-NEXT: store i8* [[FOURTEEN]], i8** [[val:%.*]]
}

