// RUN: %clang_cc1 -funknown-anytype -fsyntax-only -fdebugger-support -verify %s

extern __unknown_anytype test0;
extern __unknown_anytype test1(void);

@interface A
- (int*)getIntPtr;
- (double*)getSomePtr;
@end

@interface B
- (float*)getFloatPtr;
- (short*)getSomePtr;
@end

void test_unknown_anytype_receiver(void) {
  int *ip = [test0 getIntPtr];
  float *fp = [test1() getFloatPtr];
  double *dp = [test1() getSomePtr]; // okay: picks first method found
  [[test0 unknownMethod] otherUnknownMethod]; // expected-error{{no known method '-otherUnknownMethod'; cast the message send to the method's return type}}
  (void)(int)[[test0 unknownMethod] otherUnknownMethod];;
  [[test1() unknownMethod] otherUnknownMethod]; // expected-error{{no known method '-otherUnknownMethod'; cast the message send to the method's return type}}
  (void)(id)[[test1() unknownMethod] otherUnknownMethod];

  if ([[test0 unknownMethod] otherUnknownMethod]) { // expected-error{{no known method '-otherUnknownMethod'; cast the message send to the method's return type}}
  }
  if ([[test1() unknownMethod] otherUnknownMethod]) { // expected-error{{no known method '-otherUnknownMethod'; cast the message send to the method's return type}}
  }
}
