// RUN: %clang_cc1 -x objective-c++ -fsyntax-only -Werror -verify -Wno-objc-root-class %s
// expected-no-diagnostics
// rdar://10387088

@interface MyClass
- (void)someMethod;
@end

struct S {
  int bar(MyClass * myObject);

  int gorfbar(MyClass * myObject);

  S();
  S(MyClass *O1, MyClass *O2);
  S(MyClass *O1);

  MyClass * Obj1, *Obj2;

};

@implementation MyClass
- (void)someMethod {
    [self privateMethod];  // clang already does not warn here
}

int S::bar(MyClass * myObject) {
    [myObject privateMethod]; 
    return gorfbar(myObject);
}
- (void)privateMethod { }

int S::gorfbar(MyClass * myObject) {
    [myObject privateMethod]; 
    [myObject privateMethod1]; 
    return getMe + bar(myObject);
}

S::S(MyClass *O1, MyClass *O2) : Obj1(O1), Obj2(O2) {
    [O1 privateMethod]; 
    [O2 privateMethod1]; 
}
S::S(MyClass *O1) : Obj1(O1){ Obj2 = 0; }

S::S() {}

- (void)privateMethod1 {
  getMe = getMe+1;
}

static int getMe;

@end
