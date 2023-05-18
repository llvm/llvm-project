// Test for code completions related to Key-Value Coding and Key-Value Observing.
// Since this test is line- and column-sensitive, the run lines are at the end.

typedef signed char BOOL;

@interface NSSet
@end

@interface NSMutableSet : NSSet
@end

@interface MySet : NSMutableSet
@end

@interface NSArray
@end

@interface NSMutableArray : NSArray
@end

@interface MyArray : NSMutableArray
@end

@interface MyClass
@property int intProperty;
@property BOOL boolProperty;
@property int* intPointerProperty;

@property NSSet *setProperty;
@property NSMutableSet *mutableSetProperty;
@property MySet *mySetProperty;

@property NSArray *arrayProperty;
@property NSMutableArray *mutableArrayProperty;
@property MyArray *myArrayProperty;
@end

@implementation MyClass
- (int)intProperty { return 0; }
- (void)setIntProperty:(int)newValue { }
+ (NSSet *)keyPathsForValuesAffectingIntProperty { return 0; }
@end

@interface MySubClass : MyClass
@end

@interface MySubClass ()
@property int intProperty;
@end

@implementation MySubClass
- (int)intProperty { return 0; }
@end

// RUN: c-index-test -code-completion-at=%s:39:3 %s | FileCheck -check-prefix=CHECK-CC1 %s
// CHECK-CC1: ObjCInstanceMethodDecl:{LeftParen (}{Text void}{RightParen )}{TypedText addMutableArrayPropertyObject:}{LeftParen (}{Placeholder object-type}{Text  *}{RightParen )}{Text object} (55)
// CHECK-CC1: ObjCInstanceMethodDecl:{LeftParen (}{Text void}{RightParen )}{TypedText addMutableSetProperty:}{LeftParen (}{Text NSSet *}{RightParen )}{Text objects} (40)
// CHECK-CC1: ObjCInstanceMethodDecl:{LeftParen (}{Text NSArray *}{RightParen )}{TypedText arrayProperty} (40)
// CHECK-CC1: ObjCInstanceMethodDecl:{LeftParen (}{Text NSArray *}{RightParen )}{TypedText arrayPropertyAtIndexes:}{LeftParen (}{Text NSIndexSet *}{RightParen )}{Text indexes} (40)
// CHECK-CC1: ObjCInstanceMethodDecl:{LeftParen (}{Text BOOL}{RightParen )}{TypedText boolProperty} (40)
// CHECK-CC1: ObjCInstanceMethodDecl:{LeftParen (}{Text NSArray *}{RightParen )}{TypedText boolPropertyAtIndexes:}{LeftParen (}{Text NSIndexSet *}{RightParen )}{Text indexes} (55)
// CHECK-CC1: ObjCInstanceMethodDecl:{LeftParen (}{Text NSUInteger}{RightParen )}{TypedText countOfArrayProperty} (40)
// CHECK-CC1: ObjCInstanceMethodDecl:{LeftParen (}{Text NSUInteger}{RightParen )}{TypedText countOfBoolProperty} (55)
// CHECK-CC1: ObjCInstanceMethodDecl:{LeftParen (}{Text NSUInteger}{RightParen )}{TypedText countOfSetProperty} (40)
// CHECK-CC1: ObjCInstanceMethodDecl:{LeftParen (}{Text NSEnumerator *}{RightParen )}{TypedText enumeratorOfMutableArrayProperty} (55)
// CHECK-CC1: ObjCInstanceMethodDecl:{LeftParen (}{Text NSEnumerator *}{RightParen )}{TypedText enumeratorOfMutableSetProperty} (40)
// CHECK-CC1: ObjCInstanceMethodDecl:{LeftParen (}{Text void}{RightParen )}{TypedText getMyArrayProperty:}{LeftParen (}{Placeholder object-type}{Text  **}{RightParen )}{Text buffer}{HorizontalSpace  }{TypedText range:}{LeftParen (}{Text NSRange}{RightParen )}{Text inRange} (40)
// CHECK-CC1: ObjCInstanceMethodDecl:{LeftParen (}{Text void}{RightParen )}{TypedText getMySetProperty:}{LeftParen (}{Placeholder object-type}{Text  **}{RightParen )}{Text buffer}{HorizontalSpace  }{TypedText range:}{LeftParen (}{Text NSRange}{RightParen )}{Text inRange} (55)
// CHECK-CC1: ObjCInstanceMethodDecl:{LeftParen (}{Text void}{RightParen )}{TypedText insertIntProperty:}{LeftParen (}{Text NSArray *}{RightParen )}{Text array}{HorizontalSpace  }{TypedText atIndexes:}{LeftParen (}{Placeholder NSIndexSet *}{RightParen )}{Text indexes} (55)
// CHECK-CC1: ObjCInstanceMethodDecl:{LeftParen (}{Text void}{RightParen )}{TypedText insertMutableArrayProperty:}{LeftParen (}{Text NSArray *}{RightParen )}{Text array}{HorizontalSpace  }{TypedText atIndexes:}{LeftParen (}{Placeholder NSIndexSet *}{RightParen )}{Text indexes} (40)
// CHECK-CC1: ObjCInstanceMethodDecl:{LeftParen (}{Text void}{RightParen )}{TypedText insertObject:}{LeftParen (}{Placeholder object-type}{Text  *}{RightParen )}{Text object}{HorizontalSpace  }{TypedText inMyArrayPropertyAtIndex:}{LeftParen (}{Placeholder NSUInteger}{RightParen )}{Text index} (40)
// CHECK-CC1: ObjCInstanceMethodDecl:{LeftParen (}{Text void}{RightParen )}{TypedText insertObject:}{LeftParen (}{Placeholder object-type}{Text  *}{RightParen )}{Text object}{HorizontalSpace  }{TypedText inMySetPropertyAtIndex:}{LeftParen (}{Placeholder NSUInteger}{RightParen )}{Text index} (55)
// CHECK-CC1: ObjCInstanceMethodDecl:{LeftParen (}{Text void}{RightParen )}{TypedText intersectMySetProperty:}{LeftParen (}{Text NSSet *}{RightParen )}{Text objects} (40)
// CHECK-CC1: ObjCInstanceMethodDecl:{LeftParen (}{Text void}{RightParen )}{TypedText intersectSetProperty:}{LeftParen (}{Text NSSet *}{RightParen )}{Text objects} (55)
// CHECK-CC1: ObjCInstanceMethodDecl:{LeftParen (}{Text int *}{RightParen )}{TypedText intPointerProperty} (40)
// CHECK-CC1: ObjCInstanceMethodDecl:{LeftParen (}{Text int}{RightParen )}{TypedText intProperty} (40)
// CHECK-CC1: ObjCInstanceMethodDecl:{LeftParen (}{Text BOOL}{RightParen )}{TypedText isBoolProperty} (40)
// CHECK-CC1: ObjCInstanceMethodDecl:{LeftParen (}{Text BOOL}{RightParen )}{TypedText isIntProperty} (40)
// CHECK-CC1: ObjCInstanceMethodDecl:{LeftParen (}{Placeholder object-type}{Text  *}{RightParen )}{TypedText memberOfMyArrayProperty:}{LeftParen (}{Placeholder object-type}{Text  *}{RightParen )}{Text object} (55)
// CHECK-CC1: ObjCInstanceMethodDecl:{LeftParen (}{Placeholder object-type}{Text  *}{RightParen )}{TypedText memberOfMySetProperty:}{LeftParen (}{Placeholder object-type}{Text  *}{RightParen )}{Text object} (40)
// CHECK-CC1: ObjCInstanceMethodDecl:{LeftParen (}{Text NSMutableArray *}{RightParen )}{TypedText mutableArrayProperty} (40)
// CHECK-CC1: ObjCInstanceMethodDecl:{LeftParen (}{Text NSMutableSet *}{RightParen )}{TypedText mutableSetProperty} (40)
// CHECK-CC1: ObjCInstanceMethodDecl:{LeftParen (}{Text NSArray *}{RightParen )}{TypedText mutableSetPropertyAtIndexes:}{LeftParen (}{Text NSIndexSet *}{RightParen )}{Text indexes} (55)
// CHECK-CC1: ObjCInstanceMethodDecl:{LeftParen (}{Text MyArray *}{RightParen )}{TypedText myArrayProperty} (40)
// CHECK-CC1: ObjCInstanceMethodDecl:{LeftParen (}{Text NSArray *}{RightParen )}{TypedText myArrayPropertyAtIndexes:}{LeftParen (}{Text NSIndexSet *}{RightParen )}{Text indexes} (40)
// CHECK-CC1: ObjCInstanceMethodDecl:{LeftParen (}{Text id}{RightParen )}{TypedText objectInMutableSetPropertyAtIndex:}{LeftParen (}{Text NSUInteger}{RightParen )}{Text index} (55)
// CHECK-CC1: ObjCInstanceMethodDecl:{LeftParen (}{Text id}{RightParen )}{TypedText objectInMyArrayPropertyAtIndex:}{LeftParen (}{Text NSUInteger}{RightParen )}{Text index} (40)
// CHECK-CC1: ObjCInstanceMethodDecl:{LeftParen (}{Text void}{RightParen )}{TypedText removeMyArrayPropertyAtIndexes:}{LeftParen (}{Text NSIndexSet *}{RightParen )}{Text indexes} (40)
// CHECK-CC1: ObjCInstanceMethodDecl:{LeftParen (}{Text void}{RightParen )}{TypedText removeMySetPropertyAtIndexes:}{LeftParen (}{Text NSIndexSet *}{RightParen )}{Text indexes} (55)
// CHECK-CC1: ObjCInstanceMethodDecl:{LeftParen (}{Text void}{RightParen )}{TypedText removeObjectFromIntPropertyAtIndex:}{LeftParen (}{Text NSUInteger}{RightParen )}{Text index} (55)
// CHECK-CC1: ObjCInstanceMethodDecl:{LeftParen (}{Text void}{RightParen )}{TypedText removeObjectFromMutableArrayPropertyAtIndex:}{LeftParen (}{Text NSUInteger}{RightParen )}{Text index} (40)
// CHECK-CC1: ObjCInstanceMethodDecl:{LeftParen (}{Text void}{RightParen )}{TypedText replaceMyArrayPropertyAtIndexes:}{LeftParen (}{Placeholder NSIndexSet *}{RightParen )}{Text indexes}{HorizontalSpace  }{TypedText withMyArrayProperty:}{LeftParen (}{Text NSArray *}{RightParen )}{Text array} (40)
// CHECK-CC1: ObjCInstanceMethodDecl:{LeftParen (}{Text void}{RightParen )}{TypedText replaceMySetPropertyAtIndexes:}{LeftParen (}{Placeholder NSIndexSet *}{RightParen )}{Text indexes}{HorizontalSpace  }{TypedText withMySetProperty:}{LeftParen (}{Text NSArray *}{RightParen )}{Text array} (55)
// CHECK-CC1: ObjCInstanceMethodDecl:{LeftParen (}{Text void}{RightParen )}{TypedText replaceObjectInMyArrayPropertyAtIndex:}{LeftParen (}{Placeholder NSUInteger}{RightParen )}{Text index}{HorizontalSpace  }{TypedText withObject:}{LeftParen (}{Text id}{RightParen )}{Text object} (40)
// CHECK-CC1: ObjCInstanceMethodDecl:{LeftParen (}{Text void}{RightParen )}{TypedText replaceObjectInMySetPropertyAtIndex:}{LeftParen (}{Placeholder NSUInteger}{RightParen )}{Text index}{HorizontalSpace  }{TypedText withObject:}{LeftParen (}{Text id}{RightParen )}{Text object} (55)

// RUN: c-index-test -code-completion-at=%s:41:3 %s | FileCheck -check-prefix=CHECK-CC2 %s
// CHECK-CC2: ObjCClassMethodDecl:{LeftParen (}{Text BOOL}{RightParen )}{TypedText automaticallyNotifiesObserversOfArrayProperty} (40)
// CHECK-CC2: ObjCClassMethodDecl:{LeftParen (}{Text BOOL}{RightParen )}{TypedText automaticallyNotifiesObserversOfMutableArrayProperty} (40)
// CHECK-CC2: ObjCClassMethodDecl:{LeftParen (}{Text NSSet<NSString *> *}{RightParen )}{TypedText keyPathsForValuesAffectingMutableArrayProperty} (40)

// RUN: c-index-test -code-completion-at=%s:52:8 %s | FileCheck -check-prefix=CHECK-CC3 %s
// CHECK-CC3: ObjCInstanceMethodDecl:{TypedText countOfIntProperty} (55)
// CHECK-CC3-NEXT: ObjCInstanceMethodDecl:{TypedText intProperty} (42)
// CHECK-CC3-NEXT: ObjCInstanceMethodDecl:{TypedText isIntProperty} (40)
