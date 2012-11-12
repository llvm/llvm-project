// RUN: %clang_cc1  -fsyntax-only -fobjc-default-synthesize-properties -verify -Wno-objc-root-class %s

// rdar:// 8565343
@interface Foo  {
@private
    int _foo;
    int _foo2;
}
@property (readwrite, nonatomic) int foo, foo1, foo2, foo3;
@property (readwrite, nonatomic) int PROP;
@end

@implementation Foo

@synthesize foo = _foo;
@synthesize foo1;

- (void)setFoo:(int)value {
    _foo = foo; // expected-error {{use of undeclared identifier 'foo'}}
}

- (void)setFoo1:(int)value {
    _foo = foo1; // OK
}

- (void)setFoo2:(int)value {
    _foo = foo2; // expected-error {{use of undeclared identifier 'foo2'}}
}

- (void)setFoo3:(int)value {
    _foo = foo3;	// OK
}

@synthesize foo2 = _foo2;
@synthesize foo3;

@synthesize PROP=PROP;
- (void)setPROP:(int)value {
    PROP = value;        // OK
}

@end

