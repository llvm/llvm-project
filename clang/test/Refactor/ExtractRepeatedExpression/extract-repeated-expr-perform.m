
@interface Object {
  int ivar;
}

- (int)instanceMethod;

@property int prop;
@property void (^block)();

@end

@interface Wrapper

- (Object *)returnsObject:(int)arg;

+ (Object *)classMethodReturnsObject;

@property(class) Object *classObject;

@property Object *object;

@end

void takesClass(Wrapper *ref) {
  int x = 0;
  [[ref returnsObject: x] instanceMethod];
  int y = x;
  [ref returnsObject: x].prop;
}
// CHECK1: "Object *duplicate = [ref returnsObject:x];\n" [[@LINE-4]]:3 -> [[@LINE-4]]:3
// CHECK1-NEXT: "duplicate" [[@LINE-5]]:4 -> [[@LINE-5]]:26
// CHECK1-NEXT: "duplicate" [[@LINE-4]]:3 -> [[@LINE-4]]:25

// RUN: clang-refactor-test perform -action extract-repeated-expr-into-var -at=%s:27:4 %s | FileCheck --check-prefix=CHECK1 %s
// RUN: clang-refactor-test perform -action extract-repeated-expr-into-var -at=%s:29:3 %s | FileCheck --check-prefix=CHECK1 %s

void takesClass2(Wrapper *ref) {
  if (2)
    [[ref object] instanceMethod];
  [ref object].block();
}
// CHECK2: "Object *object = [ref object];\n" [[@LINE-4]]:3 -> [[@LINE-4]]:3
// CHECK2-NEXT: "object" [[@LINE-4]]:6 -> [[@LINE-4]]:18
// CHECK2-NEXT: "object" [[@LINE-4]]:3 -> [[@LINE-4]]:15

// RUN: clang-refactor-test perform -action extract-repeated-expr-into-var -at=%s:40:6 %s | FileCheck --check-prefix=CHECK2 %s
// RUN: clang-refactor-test perform -action extract-repeated-expr-into-var -at=%s:41:3 %s | FileCheck --check-prefix=CHECK2 %s

void takesClass3(Wrapper *ref) {
  if (ref) {
    [ref.object instanceMethod];
    ref.object.block();
  }
}
// CHECK3: "Object *object = ref.object;\n" [[@LINE-4]]:5 -> [[@LINE-4]]:5
// CHECK3-NEXT: "object" [[@LINE-5]]:6 -> [[@LINE-5]]:16
// CHECK3-NEXT: "object" [[@LINE-5]]:5 -> [[@LINE-5]]:15

// RUN: clang-refactor-test perform -action extract-repeated-expr-into-var -at=%s:52:6 %s | FileCheck --check-prefix=CHECK3 %s

void worksOnClass() {
  [Wrapper classMethodReturnsObject]->ivar = 0;
  [[Wrapper classMethodReturnsObject] instanceMethod];
}
// CHECK4: "Object *classMethodReturnsObject = [Wrapper classMethodReturnsObject];\nclassMethodReturnsObject" [[@LINE-3]]:3 -> [[@LINE-3]]:37

// CHECK4-NEXT: "classMethodReturnsObject" [[@LINE-4]]:4 -> [[@LINE-4]]:38

// RUN: clang-refactor-test perform -action extract-repeated-expr-into-var -at=%s:63:4 %s | FileCheck --check-prefix=CHECK4 %s

void worksOnClassProperty() {
  Wrapper.classObject->ivar = 0;
  Wrapper.classObject.prop = 2;
}
// CHECK5: "Object *classObject = Wrapper.classObject;\nclassObject" [[@LINE-3]]:3 -> [[@LINE-3]]:22

// CHECK5-NEXT: "classObject" [[@LINE-4]]:3 -> [[@LINE-4]]:22

// RUN: clang-refactor-test perform -action extract-repeated-expr-into-var -at=%s:73:3 %s | FileCheck --check-prefix=CHECK5 %s

#define MACROARG1(X, Y) (X)
#define MACROARG2(X) X; ref.object.prop = 0

void macroArgument(Wrapper *ref) {
  // macro-arg1: +1:13
  MACROARG1(Wrapper.classObject->ivar, 1); // MACRO-ARG1: "Object *classObject = Wrapper.classObject;\n" [[@LINE]]:3 -> [[@LINE]]:3
                                             // MACRO-ARG1-NEXT: "classObject" [[@LINE-1]]:13 -> [[@LINE-1]]:32
  MACROARG1(Wrapper.classObject.prop, 0) = 0;// MACRO-ARG1-NEXT: "classObject" [[@LINE]]:13 -> [[@LINE]]:32

  // macro-arg2: +3:13
  MACROARG2(int x = 0; ref.object->ivar); // MACRO-ARG2: "Object *object = ref.object;\n" [[@LINE]]:3 -> [[@LINE]]:3
                                          // MACRO-ARG2-NEXT: "object" [[@LINE-1]]:24 -> [[@LINE-1]]:34
  MACROARG2(ref.object.prop);// MARO-ARG2-NEXT: "object" [[@LINE]]:13 -> [[@LINE]]:23
}

// RUN: clang-refactor-test perform -action extract-repeated-expr-into-var -at=macro-arg1 %s | FileCheck --check-prefix=MACRO-ARG1 %s
// RUN: clang-refactor-test perform -action extract-repeated-expr-into-var -at=macro-arg2 %s | FileCheck --check-prefix=MACRO-ARG2 %s

@interface IVarSelf {
  Wrapper *ref;
}

@end

@implementation IVarSelf

- (void)foo {
  // ivar-self: +1:3         // IVAR-SELF: "Object *object = self->ref.object;\nobject" [[@LINE+1]]:3 -> [[@LINE+1]]:13
  ref.object.prop = 0;       // IVAR-NO-SELF: "Object *object = ref.object;\nobject" [[@LINE]]:3 -> [[@LINE]]:13
  ref.object->ivar = 1;      // IVAR: "object" [[@LINE]]:3 -> [[@LINE]]:13
  // ivar-self2: +1:3
  self->ref.object.prop = 2; // IVAR-NEXT: "object" [[@LINE]]:3 -> [[@LINE]]:19
}

@end

// RUN: clang-refactor-test perform -action extract-repeated-expr-into-var -at=ivar-self %s | FileCheck --check-prefixes=IVAR-NO-SELF,IVAR %s
// RUN: clang-refactor-test perform -action extract-repeated-expr-into-var -at=ivar-self2 %s | FileCheck --check-prefixes=IVAR-SELF,IVAR %s
