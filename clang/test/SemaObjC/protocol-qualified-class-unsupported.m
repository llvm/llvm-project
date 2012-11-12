// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

#include <stddef.h>

typedef struct objc_class *Class;
typedef struct objc_object {
    Class isa;
} *id;
id objc_getClass(const char *s);

@interface Object
+ self;
@end

@protocol Func
+ (void) class_func0;
- (void) instance_func0;
@end

@interface Derived: Object <Func>
@end

@interface Derived2: Object <Func>
@end

static void doSomething(Class <Func> unsupportedObjectType) {
  [unsupportedObjectType class_func0];
}

static void doSomethingElse(id <Func> pleaseConvertToThisType) {
  [pleaseConvertToThisType class_func0];
}

int main(int argv, char *argc[]) {
  doSomething([Derived self]);
  doSomething([Derived2 self]);
  doSomethingElse([Derived self]);
  doSomethingElse([Derived2 self]);
}

