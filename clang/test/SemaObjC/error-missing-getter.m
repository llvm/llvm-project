// RUN: %clang_cc1  -fsyntax-only -verify %s

@interface Subclass 
{
    int setterOnly;
}
- (void) setSetterOnly : (int) arg;
@end

int func (int arg, Subclass *x) {
    if (x.setterOnly) { // expected-error {{no getter method for read from property}}
      x.setterOnly = 1;
    }
    func(x.setterOnly + 1, x); // expected-error {{no getter method for read from property}}
    int i = x.setterOnly + 1;  // expected-error {{no getter method for read from property}}
    return x.setterOnly + 1;   // expected-error {{no getter method for read from property}}
}

@interface TestClass 
+ (void) setSetterOnly : (int) arg;
@end

int func2 (int arg) {
    if (TestClass.setterOnly) { // expected-error {{no getter method for read from property}}
      TestClass.setterOnly = 1;
    }
    func(TestClass.setterOnly + 1, x); // expected-error {{no getter method for read from property}} \
                                       // expected-error {{use of undeclared identifier 'x'}}
    int i = TestClass.setterOnly + 1;  // expected-error {{no getter method for read from property}}
    return TestClass.setterOnly + 1;   // expected-error {{no getter method for read from property}}
}

@interface Sub : Subclass
- (int) func3;
@end
@implementation Sub
- (int) func3 {
	return super.setterOnly; // expected-error {{no getter method for read from property}}
}
@end
