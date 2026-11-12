// RUN: %clang_cc1 -fsyntax-only -verify -Wno-objc-root-class %s -Wselector -include %S/Inputs/chain-selectors1.h -include %S/Inputs/chain-selectors2.h

// RUN: %clang_cc1 -x objective-c -Wno-objc-root-class -emit-pch -o %t1 %S/Inputs/chain-selectors1.h
// RUN: %clang_cc1 -x objective-c -Wno-objc-root-class -emit-pch -o %t2 %S/Inputs/chain-selectors2.h -include-pch %t1
// RUN: %clang_cc1 -fsyntax-only -verify -Wno-objc-root-class %s -Wselector -include-pch %t2

@implementation X
-(void)f {}
-(void)f2 {}
-(void)g: (int)p {}
-(void)h: (int)p1 foo: (int)p2 {}
@end

void bar(void) {
  id a = 0;
  [a nothing]; // expected-warning {{method '-nothing' not found}}
  [a f];
  // FIXME: Can't verify notes in headers
  //[a f2];

  (void)@selector(x); // expected-warning {{no method with selector 'x' is implemented in this translation unit}}
  (void)@selector(y); // expected-warning {{no method with selector 'y' is implemented in this translation unit}}
  (void)@selector(e); // expected-warning {{no method with selector 'e' is implemented in this translation unit}}
}

@implementation X (Blah)
- (void)test_Blah {
  [self blah_method];
}

- (void)blah_method { }
@end

@implementation X (Blarg)
- (void)test_Blarg {
  [self blarg_method];
}

- (void)blarg_method { }
@end
