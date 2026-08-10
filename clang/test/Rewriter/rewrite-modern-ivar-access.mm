// RUN: %clang_cc1 -fblocks -rewrite-objc -fms-extensions %s -o %t-rw.cpp
// RUN: %clang_cc1 -Werror -fsyntax-only -Wno-address-of-temporary -Wno-c++11-narrowing -std=c++11 -D"Class=void*" -D"id=void*" -D"SEL=void*" -U__declspec -D"__declspec(X)=" %t-rw.cpp

struct OUTSIDE {
  int i_OUTSIDE;
  double d_OUTSIDE;
};


@interface I1 {
@protected
  struct OUTSIDE ivar_I1;

  struct INNER_I1 {
    int i_INNER_I1;
    double d_INNER_I1;
  };

  struct INNER_I1 ivar_I2;

  struct OUTSIDE ivar_I3;

  struct {
    int i_noname;
    double d_noname;
  } NONAME_I4;

  struct {
    int i_noname;
    double d_noname;
  } NONAME_I5;
}
@end

@implementation I1
- (void) I1_Meth {
  ivar_I1.i_OUTSIDE = 0;

  ivar_I2.i_INNER_I1 = 1;

  ivar_I3.i_OUTSIDE = 2;

  NONAME_I4.i_noname = 3;

  NONAME_I5.i_noname = 4;
}
@end

@interface INTF2 {
@protected
  struct OUTSIDE ivar_INTF2;

  struct {
    int i_noname;
    double d_noname;
  } NONAME_INTF4;


  struct OUTSIDE ivar_INTF3;

  struct INNER_I1 ivar_INTF4;

  struct {
    int i_noname;
    double d_noname;
  } NONAME_INTF5;

  struct INNER_INTF2 {
    int i_INNER_INTF2;
    double d_INNER_INTF2;
  };

  struct INNER_INTF2 ivar_INTF6, ivar_INTF7;

  struct INNER_INTF3 {
    int i;
  } X1,X2,X3;

}
@end

@implementation INTF2
- (void) I2_Meth {
  ivar_INTF2.i_OUTSIDE = 0;

  ivar_INTF4.i_INNER_I1 = 1;

  ivar_INTF3.i_OUTSIDE = 2;

  NONAME_INTF4.i_noname = 3;

  NONAME_INTF5.i_noname = 4;
  ivar_INTF6.i_INNER_INTF2 = 5;
  ivar_INTF7.i_INNER_INTF2 = 5;
  X1.i = X2.i = X3.i = 1;
}
@end

