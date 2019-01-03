// RUN: clang-refactor-test perform -action extract -selected=prop -selected=imp-prop -selected=class-prop -selected=class-prop2 -selected=class-method %s | FileCheck %s

@interface NSObject
@end

@interface Array<Element> : NSObject

@property Element prop;

- (Element)get;

@property (class) Array<Element> *classProp;

+ (Element *)classGet;

@end

void foo(Array<NSObject *> *objects) {
// prop-begin: +1:3
  objects.prop;
// prop-end: -1:15
// CHECK: "static NSObject *extracted(Array<NSObject *> *objects) {\nreturn objects.prop;\n}\n\n"
// imp-prop-begin: +1:3
  objects.get;
// imp-prop-end: -1:14
// CHECK: "static NSObject *extracted(Array<NSObject *> *objects) {\nreturn objects.get;\n}\n\n"
// class-prop-begin: +1:3
  Array.classProp;
// class-prop-end: -1:30
// CHECK: "static Array *extracted() {\nreturn Array.classProp;\n}\n\n"
  typedef Array<NSObject *> ObjectArray;
// class-prop2-begin: +1:3
  [ObjectArray classProp];
// class-prop2-end: -1:26
// CHECK: "static Array<NSObject *> *extracted() {\nreturn [ObjectArray classProp];\n}\n\n"
// class-method-begin: +1:3
  [ObjectArray classGet];
// class-method-end: -1:25
// CHECK: "static NSObject **extracted() {\nreturn [ObjectArray classGet];\n}\n\n"
}
