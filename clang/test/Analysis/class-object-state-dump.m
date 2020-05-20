// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection \
// RUN:   -verify %s 2>&1 | FileCheck %s

// expected-no-diagnostics

void clang_analyzer_printState();

@interface NSObject {
}
+ (id)alloc;
+ (Class)class;
- (id)init;
- (Class)class;
@end

@interface Parent : NSObject
@end
@interface Child : Parent
@end

void foo(id A, id B);

@implementation Child
+ (void)test {
  id ClassAsID = [self class];
  id Object = [[ClassAsID alloc] init];
  Class TheSameClass = [Object class];

  clang_analyzer_printState();
  // CHECK:      "class_object_types": [
  // CHECK-NEXT:   { "symbol": "conj_$[[#]]{Class, LC[[#]], S[[#]], #[[#]]}", "dyn_type": "Child", "sub_classable": true },
  // CHECK-NEXT:   { "symbol": "conj_$[[#]]{Class, LC[[#]], S[[#]], #[[#]]}", "dyn_type": "Child", "sub_classable": true }
  // CHECK-NEXT: ]

  // Let's make sure that the information is not GC'd away.
  foo(ClassAsID, TheSameClass);
}
@end
