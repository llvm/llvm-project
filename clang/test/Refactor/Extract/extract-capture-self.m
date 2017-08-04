
@interface AClass {
  int ivar1;
}

@property int prop;

- (int)instanceMethod;
+ (int)classMethod;

@end

@implementation AClass {
  int ivar2;
}

- (int)instanceMethod {
  ivar2 = 0;
  ivar1 = 0;
  self->ivar2 = 0;
  self.prop = 0;
  int x = self->ivar1;
  int y = self.prop;
  [self instanceMethod];
  [AClass classMethod];
  return 0;
}
// CHECK1: (AClass *object) {\nobject->ivar2 = 0;\n}
// CHECK1: (AClass *object) {\nobject->ivar2 = 0;\n}
// CHECK1: (AClass *object) {\nobject.prop = 0;\n}\n\n"
// CHECK1: (AClass *object) {\nint y = object.prop;\n}
// CHECK1: (AClass *object) {\nreturn [object instanceMethod];\n}
// CHECK1: (AClass *object) {\nobject->ivar2 = 0;\n  object->ivar1 = 0;\n  object->ivar2 = 0;\n  object.prop = 0;\n  int x = object->ivar1;\n  int y = object.prop;\n  [object instanceMethod];\n  [AClass classMethod];\n}\n\n"
// CHECK1: () {\nreturn [AClass classMethod];\n}\n\n"

// RUN: clang-refactor-test perform -action extract -selected=%s:18:3-18:12 -selected=%s:20:3-20:18 -selected=%s:21:3-21:16 -selected=%s:23:3-23:20 -selected=%s:24:3-24:24 -selected=%s:18:3-25:23 -selected=%s:25:3-25:23 %s | FileCheck --check-prefix=CHECK1 %s

+ (int)classMethod {
  int x = self.classMethod;
  [self classMethod];
}

// CHECK2: () {\nint x = AClass.classMethod;\n}
// CHECK2: () {\nreturn [AClass classMethod];\n}

// RUN: clang-refactor-test perform -action extract -selected=%s:39:3-39:27 -selected=%s:40:3-40:21 %s | FileCheck --check-prefix=CHECK2 %s

- (void)rhsSelfCaptureAndRewrite:(AClass *)i { // CHECK3: "static void extracted(AClass *object, AClass *i) {\ni.prop= object.prop;\n}\n\n"
// rhs-prop-begin: +1:3
  i.prop= self.prop;
// rhs-prop-end: -1:21
}

// RUN: clang-refactor-test perform -action extract -selected=rhs-prop %s | FileCheck --check-prefix=CHECK3 %s

@end
