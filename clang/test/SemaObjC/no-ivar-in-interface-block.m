// RUN: %clang_cc1  -fsyntax-only -verify -Wno-objc-root-class -Wobjc-interface-ivars %s

@interface I
{
  @protected  int P_IVAR; // expected-warning {{declaration of instance variables in the interface is deprecated}}

  @public     int PU_IVAR; // expected-warning {{declaration of instance variables in the interface is deprecated}}

  @private    int PRV_IVAR; // expected-warning {{declaration of instance variables in the interface is deprecated}}
}
@end

@interface I()
{
  int I1;
  int I2;
}
@end

@interface I()
{
  int I3, I4;
}
@end

@implementation I
{
  int I5;
  int I6;
}
@end 
