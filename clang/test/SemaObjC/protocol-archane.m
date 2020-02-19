// RUN: %clang_cc1 -fsyntax-only -verify -Wno-pointer-to-int-cast -Wno-objc-root-class %s
// rdar://5986251

@protocol SomeProtocol
- (void) bar;
@end

void bar();
void foo(id x) {
  bar((short<SomeProtocol>)x); // expected-error {{expected ')'}} expected-note {{to match this '('}}
  bar((<SomeProtocol>)x);      // expected-warning {{protocol has no object type specified; defaults to qualified 'id'}}

  [(<SomeProtocol>)x bar];      // expected-warning {{protocol has no object type specified; defaults to qualified 'id'}}
}

@protocol MyProtocol
- (void)doSomething;
@end

@interface MyClass
- (void)m1:(id <MyProtocol> const)arg1;

// FIXME: provide a better diagnostic (no typedef).
- (void)m2:(id <MyProtocol> short)arg1; // expected-error {{'short type-name' is invalid}}
@end

typedef int NotAnObjCObjectType;

// GCC doesn't diagnose this.
NotAnObjCObjectType <SomeProtocol> *obj; // expected-error {{invalid protocol qualifiers on non-ObjC type}}

typedef struct objc_class *Class;

Class <SomeProtocol> UnfortunateGCCExtension;

// rdar://10238337
@protocol Broken @end
@interface Crash @end
@implementation Crash
- (void)crashWith:(<Broken>)a { // expected-warning {{protocol has no object type specified; defaults to qualified 'id'}}
}
@end

typedef <SomeProtocol> id TwoTypeSpecs; // expected-warning{{no object type specified}}
// expected-error@-1{{typedef redefinition with different types ('id<SomeProtocol>' vs 'id')}}
// expected-error@-2{{expected ';' after top level declarator}}
