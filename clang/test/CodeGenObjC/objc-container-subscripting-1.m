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

int main() {
  NSMutableArray *array;
  id val;

  id oldObject = array[10];
// CHECK: [[ARR:%.*]] = load {{%.*}} [[array:%.*]], align 8
// CHECK-NEXT: [[SEL:%.*]] = load i8** @"\01L_OBJC_SELECTOR_REFERENCES_"
// CHECK-NEXT: [[ARRC:%.*]] = bitcast {{%.*}} [[ARR]] to i8*
// CHECK-NEXT: [[CALL:%.*]] = call i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8* (i8*, i8*, i32)*)(i8* [[ARRC]], i8* [[SEL]], i32 10)
// CHECK-NEXT: store i8* [[CALL]], i8** [[OLDOBJ:%.*]], align 8

  val = (array[10] = oldObject);
// CHECK: [[THREE:%.*]] = load {{%.*}} [[array:%.*]], align 8
// CHECK-NEXT: [[FOUR:%.*]] = load i8** [[oldObject:%.*]], align 8
// CHECK-NEXT: [[FIVE:%.*]] = load i8** @"\01L_OBJC_SELECTOR_REFERENCES_2"
// CHECK-NEXT: [[SIX:%.*]] = bitcast {{%.*}} [[THREE]] to i8*
// CHECK-NEXT: call void bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to void (i8*, i8*, i8*, i32)*)(i8* [[SIX]], i8* [[FIVE]], i8* [[FOUR]], i32 10)
// CHECK-NEXT: store i8* [[FOUR]], i8** [[val:%.*]]

  NSMutableDictionary *dictionary;
  id key;
  id newObject;
  oldObject = dictionary[key];
// CHECK:  [[SEVEN:%.*]] = load {{%.*}} [[DICTIONARY:%.*]], align 8
// CHECK-NEXT:  [[EIGHT:%.*]] = load i8** [[KEY:%.*]], align 8
// CHECK-NEXT:  [[TEN:%.*]] = load i8** @"\01L_OBJC_SELECTOR_REFERENCES_4"
// CHECK-NEXT:  [[ELEVEN:%.*]] = bitcast {{%.*}} [[SEVEN]] to i8*
// CHECK-NEXT:  [[CALL1:%.*]] = call i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8* (i8*, i8*, i8*)*)(i8* [[ELEVEN]], i8* [[TEN]], i8* [[EIGHT]])
// CHECK-NEXT:  store i8* [[CALL1]], i8** [[oldObject:%.*]], align 8


  val = (dictionary[key] = newObject);
// CHECK: [[TWELVE:%.*]] = load {{%.*}} [[DICTIONARY]], align 8
// CHECK-NEXT:  [[THIRTEEN:%.*]] = load i8** [[KEY]], align 8
// CHECK-NEXT:  [[FOURTEEN:%.*]] = load i8** [[NEWOBJECT:%.*]], align 8
// CHECK-NEXT:  [[SIXTEEN:%.*]] = load i8** @"\01L_OBJC_SELECTOR_REFERENCES_6"
// CHECK-NEXT:  [[SEVENTEEN:%.*]] = bitcast {{%.*}} [[TWELVE]] to i8*
// CHECK-NEXT:  call void bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to void (i8*, i8*, i8*, i8*)*)(i8* [[SEVENTEEN]], i8* [[SIXTEEN]], i8* [[FOURTEEN]], i8* [[THIRTEEN]])
// CHECK-NEXT: store i8* [[FOURTEEN]], i8** [[val:%.*]]
}

