// RUN: %clang_cc1  -fsyntax-only -Wno-deprecated-declarations -verify -Wno-objc-root-class %s

@interface Foo 
- (id)test:(id)one, id two;
- (id)bad:(id)one, id two, double three;
@end

@implementation Foo
- (id)test:(id )one, id two {return two; } 
- (id)bad:(id)one, id two, double three { return two; }
@end


int main(void) {
	Foo *foo;
	[foo test:@"One", @"Two"];
	[foo bad:@"One", @"Two"]; // expected-error {{too few arguments to method call}}
	[foo bad:@"One", @"Two", 3.14];
	[foo bad:@"One", @"Two", 3.14, @"Two"]; // expected-error {{too many arguments to method call}}
}
