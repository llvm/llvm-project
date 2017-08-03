// RUN: clang-refactor-test perform -action extract -selected=super-msg -selected=super-prop -selected=super-self -selected=super-class-msg -selected=super-class-prop %s | FileCheck %s

@interface BaseClass

@property int prop;

- (int)instanceMethod;
+ (void)classMethod;

@property(class) int classProp;

@end

@interface SubClass: BaseClass

@end

@implementation SubClass

- (void)method {
  // super-msg-begin: +1:1
  [super instanceMethod];  // CHECK: extracted(BaseClass *superObject) {\n[superObject instanceMethod];\n}
  // super-msg-end: +0:1   // CHECK: extracted(super.self)
  // super-prop-begin: +1:11
  int x = super.prop;      // CHECK: extracted(BaseClass *superObject) {\nreturn superObject.prop;\n}
  // super-prop-end: -1:21 // CHECK: extracted(super.self)
  // super-self-begin: +1:1
  int y = self.prop;      // CHECK: extracted(SubClass *object, BaseClass *superObject) {\nint y = object.prop;\n  int z = superObject.prop;\n}
  int z = super.prop;     // CHECK: extracted(self, super.self);
  // super-self-end: +0:1
}

+ (void)classMethod {
  // super-class-msg-begin: +1:1
  [super classMethod];         // CHECK: extracted() {\n[BaseClass classMethod];\n}
  // super-class-msg-end: +0:1 // CHECK: extracted()
  // super-class-prop-begin: +1:9
  (void)super.classProp;       // CHECK: extracted() {\nreturn BaseClass.classProp;\n}
  // super-class-prop-end: -1:24 // CHECK: extracted()
}

@end
