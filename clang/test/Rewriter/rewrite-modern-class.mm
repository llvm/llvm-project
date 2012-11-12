// RUN: %clang_cc1 -x objective-c++ -Wno-return-type -fblocks -fms-extensions -rewrite-objc %s -o %t-rw.cpp
// RUN: %clang_cc1 -fsyntax-only -fblocks -Wno-address-of-temporary -D"Class=void*" -D"id=void*" -D"SEL=void*" -D"__declspec(X)=" %t-rw.cpp

@protocol PROTO @end

@interface empty_root @end

@interface root_with_ivars
{
  id ROOT_IVAR;
  id ROOT1_IVAR;
}
@end

@interface MAXIMAL : root_with_ivars<PROTO>
{
  double D_IVAR;
  double D_PROPERTY;
}
- (void) V_METH;
@end

@implementation MAXIMAL
- (void) V_METH {}
@end
//=========================================
@interface empty_class @end

@implementation empty_class @end
//=========================================
@interface class_empty_root : empty_root @end

@implementation class_empty_root @end
//=========================================
@interface class_with_ivars : empty_root
{
  int class_with_ivars_IVAR;
}
@end

@implementation class_with_ivars @end
//=========================================
@interface class_has_no_ivar : root_with_ivars @end

@implementation class_has_no_ivar @end

//============================class needs to be synthesized here=====================
@interface SUPER  {
@public
  double divar;
  SUPER *p_super;
}
@end

@interface INTF @end

@implementation INTF  
- (SUPER *) Meth : (SUPER *)arg { 
  return arg->p_super; 
}
@end

@class FORM_CLASS;
@interface INTF_DECL  {
}
@end

double Meth(INTF_DECL *p, FORM_CLASS *f) {
  return 1.34;
}
