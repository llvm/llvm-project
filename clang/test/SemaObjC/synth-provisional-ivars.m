// RUN: %clang_cc1 -fsyntax-only -fobjc-default-synthesize-properties -verify -Wno-objc-root-class %s

int bar;

@interface I
{
    int _bar;
}
@property int PROP;
@property int PROP1;
@property int PROP2;
@property int PROP3;
@property int PROP4;

@property int bar;
@property int bar1;

@end

@implementation I
- (int) Meth { return _PROP; }

@dynamic PROP1;
- (int) Meth1 { return PROP1; }  // expected-error {{use of undeclared identifier 'PROP1'}}

- (int) Meth2 { return PROP2; }  // expected-error {{use of undeclared identifier 'PROP2'}}
@dynamic PROP2;

- (int) Meth3 { return PROP3; }  // expected-error {{use of undeclared identifier 'PROP3'}}
@synthesize PROP3=IVAR;

- (int) Meth4 { return PROP4; }
@synthesize PROP4=PROP4; // expected-note 4 {{'PROP4' declared here}}

- (int) Meth5 { return bar; }
@synthesize bar = _bar;

- (int) Meth6 { return _bar1; }

@end

@implementation I(CAT)
- (int) Meth { return PROP1; }  // expected-error {{use of undeclared identifier 'PROP1'}}
@end

@implementation I(r8251648)
- (int) Meth1: (int) bar {
  return bar;
}
@end
