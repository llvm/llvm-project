// RUN: %clang_cc1 -fsyntax-only -verify %s
// Make sure we don't crash in TreeTransform<Derived>::TransformObjCObjectType.

@protocol P1
@end

template <class T1><P1> foo1(T1) { // expected-warning {{protocol has no object type specified; defaults to qualified 'id'}}
  foo1(0);
}
