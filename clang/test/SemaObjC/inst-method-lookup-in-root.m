// RUN: %clang_cc1  -fsyntax-only -verify %s

@protocol P
- (id) inst_in_proto;
@end

@interface Object <P>
- (id) inst_in_root;
@end

@interface Base
@end

@interface Derived: Base
- (id)starboard;
@end

void foo(void) {
  Class receiver;

  [Derived starboard]; // expected-warning {{method '+starboard' not found}}

  [receiver starboard]; // expected-warning {{instance method 'starboard' is being used on 'Class'}}
  [receiver inst_in_root]; // Ok!
  [receiver inst_in_proto]; // Ok!
}

