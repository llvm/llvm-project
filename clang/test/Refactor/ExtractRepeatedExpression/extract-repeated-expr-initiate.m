
@interface Base

@end

@interface Object: Base {
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

void test(Wrapper *wrapper) {
  [[wrapper returnsObject: 42] instanceMethod];
  [[wrapper returnsObject: 42] prop];
// CHECK1: Initiated the 'extract-repeated-expr-into-var' action at [[@LINE-2]]:4
// CHECK2: Initiated the 'extract-repeated-expr-into-var' action at [[@LINE-2]]:4

  wrapper.object.prop;
  [wrapper.object instanceMethod];
// CHECK3: Initiated the 'extract-repeated-expr-into-var' action at [[@LINE-2]]:3
// CHECK4: Initiated the 'extract-repeated-expr-into-var' action at [[@LINE-2]]:4

  [[wrapper object] block];
  [[wrapper object] instanceMethod];
// CHECK5: Initiated the 'extract-repeated-expr-into-var' action at [[@LINE-2]]:4
// CHECK6: Initiated the 'extract-repeated-expr-into-var' action at [[@LINE-2]]:4

  [[Wrapper classMethodReturnsObject] instanceMethod];
  [[Wrapper classMethodReturnsObject] prop];
// CHECK7: Initiated the 'extract-repeated-expr-into-var' action at [[@LINE-2]]:4
// CHECK8: Initiated the 'extract-repeated-expr-into-var' action at [[@LINE-2]]:4

  Wrapper.classObject.prop;
  if (1)
    [Wrapper.classObject instanceMethod];
// CHECK9: Initiated the 'extract-repeated-expr-into-var' action at [[@LINE-3]]:3
// CHECK10: Initiated the 'extract-repeated-expr-into-var' action at [[@LINE-2]]:6
}

// RUN: clang-refactor-test initiate -action extract-repeated-expr-into-var -in=%s:30:4-31 %s -fblocks | FileCheck --check-prefix=CHECK1 %s
// RUN: clang-refactor-test initiate -action extract-repeated-expr-into-var -in=%s:31:4-31 %s -fblocks | FileCheck --check-prefix=CHECK2 %s
// RUN: not clang-refactor-test initiate -action extract-repeated-expr-into-var -in=%s:30:1-3 -in=%s:30:32-48 %s 2>&1 | FileCheck --check-prefix=CHECK-NO %s

// RUN: clang-refactor-test initiate -action extract-repeated-expr-into-var -in=%s:35:3-17 %s -fblocks | FileCheck --check-prefix=CHECK3 %s
// RUN: clang-refactor-test initiate -action extract-repeated-expr-into-var -in=%s:36:4-18 %s -fblocks | FileCheck --check-prefix=CHECK4 %s
// RUN: not clang-refactor-test initiate -action extract-repeated-expr-into-var -in=%s:35:18-23 %s 2>&1 | FileCheck --check-prefix=CHECK-NO %s

// RUN: clang-refactor-test initiate -action extract-repeated-expr-into-var -in=%s:40:4-20 %s -fblocks | FileCheck --check-prefix=CHECK5 %s
// RUN: clang-refactor-test initiate -action extract-repeated-expr-into-var -in=%s:41:4-20 %s -fblocks | FileCheck --check-prefix=CHECK6 %s
// RUN: not clang-refactor-test initiate -action extract-repeated-expr-into-var -in=%s:40:21-28 %s 2>&1 | FileCheck --check-prefix=CHECK-NO %s

// RUN: clang-refactor-test initiate -action extract-repeated-expr-into-var -in=%s:45:4-38 %s -fblocks | FileCheck --check-prefix=CHECK7 %s
// RUN: clang-refactor-test initiate -action extract-repeated-expr-into-var -in=%s:46:4-38 %s -fblocks | FileCheck --check-prefix=CHECK8 %s
// RUN: not clang-refactor-test initiate -action extract-repeated-expr-into-var -in=%s:45:39-55 %s 2>&1 | FileCheck --check-prefix=CHECK-NO %s

// RUN: clang-refactor-test initiate -action extract-repeated-expr-into-var -in=%s:50:3-22 %s -fblocks | FileCheck --check-prefix=CHECK9 %s
// RUN: clang-refactor-test initiate -action extract-repeated-expr-into-var -in=%s:52:6-25 %s -fblocks | FileCheck --check-prefix=CHECK10 %s
// RUN: not clang-refactor-test initiate -action extract-repeated-expr-into-var -in=%s:50:23-28 -in=%s:51:1-9 %s 2>&1 | FileCheck --check-prefix=CHECK-NO %s

// CHECK-NO: Failed to initiate the refactoring action!

void testInvalidMethod(Wrapper *ref) {
  if (2)
    [[ref classObject] instanceMethod];
  [ref classObject].block();
}
// RUN: not clang-refactor-test initiate -action extract-repeated-expr-into-var -in=%s:81:6-23 %s 2>&1 | FileCheck --check-prefix=CHECK-NO %s

@interface ImplicitPropertyWithoutGetter
- (void) setValue: (int) value;
@end
void implicitPropertyWithoutGetter(ImplicitPropertyWithoutGetter *x) {
  x.value = 0;
  x.value = 1;
}

// RUN: not clang-refactor-test initiate -action extract-repeated-expr-into-var -at=%s:90:3 %s 2>&1 | FileCheck --check-prefix=CHECK-NO %s

// Prohibit ininiation in macros:

#define MACROREF(X) X.object

void prohibitMacroExpr(Wrapper *wrapper) {
  // macro-prohibited: +1:3
  wrapper.object.prop = 0;
  MACROREF(wrapper).prop = 1;
}

// RUN: not clang-refactor-test initiate -action extract-repeated-expr-into-var -at=macro-prohibited %s 2>&1 | FileCheck --check-prefix=CHECK-NO %s
