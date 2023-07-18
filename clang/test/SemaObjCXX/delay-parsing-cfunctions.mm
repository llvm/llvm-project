// RUN: %clang_cc1 -x objective-c++ -std=c++11 -fsyntax-only -Werror -verify -Wno-objc-root-class %s
// expected-no-diagnostics
// rdar://10387088

struct X {
X();
void SortWithCollator();
};

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

- (void)privateMethod1 {
  getMe = getMe+1;
}

static int getMe;

static int test() {
  return 0;
}

int x{17};

X::X() = default;
void X::SortWithCollator() {}
// pr13418
namespace {
     int CurrentTabId() {return 0;}
}
@end
