// RUN: %clang_cc1 -fsyntax-only -verify %s

@interface B
+(int) classGetter;
-(int) getter;
@end

@interface A : B
@end

@implementation A
+(int) classGetter {
  return 0;
}

+(int) classGetter2 {
  return super.classGetter;
}

-(void) method {
  int x = super.getter;
}
@end

void f0(void) {
  // FIXME: not implemented yet.
  //int l1 = A.classGetter;
  int l2 = [A classGetter2];
}

// rdar://13349296
__attribute__((objc_root_class)) @interface ClassBase 
@property (nonatomic, retain) ClassBase * foo; // expected-note {{property declared here}}
@end

@implementation ClassBase 
- (void) Meth:(ClassBase*)foo {
  super.foo = foo; // expected-error {{'ClassBase' cannot use 'super' because it is a root class}}
  [super setFoo:foo]; // expected-error {{'ClassBase' cannot use 'super' because it is a root class}}
}
@end

@interface ClassDerived : ClassBase 
@property (nonatomic, retain) ClassDerived * foo; // expected-warning {{auto property synthesis will not synthesize property 'foo'; it will be implemented by its superclass}}
@end

@implementation ClassDerived // expected-note {{detected while default synthesizing properties in class implementation}}
- (void) Meth:(ClassBase*)foo {
  super.foo = foo; // must work with no warning
  [super setFoo:foo]; // works with no warning
}
@end

@implementation IFaceNotFound (Foo) // expected-error {{cannot find interface declaration for 'IFaceNotFound'}}
-(int) foo {
  return super.foo; // expected-error {{expected identifier or '('}}
}
@end
