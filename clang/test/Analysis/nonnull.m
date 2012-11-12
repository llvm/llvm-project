// RUN: %clang_cc1 -analyze -analyzer-checker=core -verify %s

@interface MyObject
- (void)takePointer:(void *)ptr __attribute__((nonnull(1)));
@end

void testNonNullMethod(int *p, MyObject *obj) {
  if (p)
    return;
  [obj takePointer:p]; // expected-warning{{nonnull}}
}


@interface Subclass : MyObject
// [[nonnull]] is an inherited attribute.
- (void)takePointer:(void *)ptr;
@end

void testSubclass(int *p, Subclass *obj) {
  if (p)
    return;
  [obj takePointer:p]; // expected-warning{{nonnull}}
}
