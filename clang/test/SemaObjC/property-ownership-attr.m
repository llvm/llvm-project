// RUN: %clang_cc1 -fsyntax-only -verify %s 
// rdar://15014468

@protocol P
  @property(readonly) id z;
@end

@interface Foo
  @property (readonly) id x;
@end

@interface MutableFoo : Foo
  @property (copy) id x;
@end

@interface Foo (Cat) <P>
@property (copy) id  z; // expected-warning {{'copy' attribute on property 'z' does not match the property inherited from 'P'}}
@end

