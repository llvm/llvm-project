// RUN: %clang_cc1  -fsyntax-only -Werror -verify -Wno-objc-root-class -Wno-strict-prototypes %s
// expected-no-diagnostics
// rdar://10387088

@interface MyClass
- (void)someMethod;
@end

@implementation MyClass
- (void)someMethod {
    [self privateMethod];  // clang already does not warn here
}

int bar(MyClass * myObject) {
    [myObject privateMethod]; 
    return gorfbar(myObject);
}
- (void)privateMethod { }

int gorfbar(MyClass * myObject) {
    [myObject privateMethod]; 
    [myObject privateMethod1]; 
    return getMe + bar(myObject);
}

int KR(myObject)
MyClass * myObject;
{
    [myObject privateMethod];
    [myObject privateMethod1];
    return getMe + bar(myObject);
}

- (void)privateMethod1 {
  getMe = getMe+1;
}

static int getMe;

static int test(void) {
  return 0;
}

@end
