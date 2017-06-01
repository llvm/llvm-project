// RUN: %clang_cc1 -x objective-c++ -Wno-return-type -fblocks -fms-extensions -rewrite-objc %s -o %t-rw.cpp
// RUN: %clang_cc1 -fsyntax-only -std=gnu++98 -fblocks -Wno-address-of-temporary -D"Class=void*" -D"id=void*" -D"SEL=void*" -D"__declspec(X)=" %t-rw.cpp

@protocol P @end
@protocol P1 @end
@interface INTF
{
  id CLASS_IVAR;
  id<P, P1> Q_IVAR;

  void (^_block)(id<P>);
  void (*_fptr)(void (^_block)(id<P>));
  char CLASS_EXT_IVAR;
  id<P, P1> (^ext_block)(id<P>, INTF<P,P1>*, INTF*);
  id IMPL_IVAR;
  double D_IMPL_IVAR;
  INTF<P> *(*imp_fptr)(void (^_block)(id<P>, INTF<P,P1>*));
  id arr[100];
}
@end

@implementation INTF @end

@interface MISC_INTF
{
  id CLASS_IVAR;
  id<P, P1> Q_IVAR;

  void (^_block)(id<P>);
  void (*_fptr)(void (^_block)(id<P>));
  unsigned int BF : 8;
}
@end

@interface MISC_INTF()
{
  char CLASS_EXT_IVAR;
  id<P, P1> (^ext_block)(id<P>, MISC_INTF<P,P1>*, MISC_INTF*);
}
@end

@interface MISC_INTF() {
  int II1;
  double DD1; }
@end

@interface MISC_INTF() { int II2; double DD2; }
@end

@interface MISC_INTF() { int II3; 
  double DD3; }
@end

@interface MISC_INTF() { int II4; double DD4; 
}
@end

@implementation MISC_INTF
{
  id IMPL_IVAR;
  double D_IMPL_IVAR;
  MISC_INTF<P> *(*imp_fptr)(void (^_block)(id<P>, MISC_INTF<P,P1>*));
}
@end
