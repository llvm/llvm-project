// RUN: %clang_cc1 -triple x86_64-apple-macosx10.13.0 -fobjc-runtime=macosx-10.13.0 -fblocks -fobjc-arc %s -verify
// RUN: %clang_cc1 -triple x86_64-apple-macosx10.13.0 -fobjc-runtime=macosx-10.13.0 -fblocks -fobjc-arc -xobjective-c++ %s -verify

#define EXT_RET __attribute__((objc_externally_retained))

@interface ObjCTy
@end

void test1(void) {
  EXT_RET int a; // expected-warning{{'objc_externally_retained' can only be applied to}}
  EXT_RET __weak ObjCTy *b; // expected-warning{{'objc_externally_retained' can only be applied to}}
  EXT_RET __weak int (^c)(void); // expected-warning{{'objc_externally_retained' can only be applied to}}

  EXT_RET int (^d)(void) = ^{return 0;};
  EXT_RET ObjCTy *e = 0;
  EXT_RET __strong ObjCTy *f = 0;

  e = 0; // expected-error{{variable declared with 'objc_externally_retained' cannot be modified in ARC}}
  f = 0; // expected-error{{variable declared with 'objc_externally_retained' cannot be modified in ARC}}
  d = ^{ return 0; }; // expected-error{{variable declared with 'objc_externally_retained' cannot be modified in ARC}}
}

void test2(ObjCTy *a);

void test2(ObjCTy *a) EXT_RET {
  a = 0; // expected-error{{variable declared with 'objc_externally_retained' cannot be modified in ARC}}
}

EXT_RET ObjCTy *test3; // expected-warning{{'objc_externally_retained' can only be applied to}}

@interface X // expected-warning{{defined without specifying a base class}} expected-note{{add a super class}}
-(void)m: (ObjCTy *) p;
@end
@implementation X
-(void)m: (ObjCTy *) p EXT_RET {
  p = 0; // expected-error{{variable declared with 'objc_externally_retained' cannot be modified in ARC}}
}
@end

void test4(void) {
  __attribute__((objc_externally_retained(0))) ObjCTy *a; // expected-error{{'objc_externally_retained' attribute takes no arguments}}
}

void test5(ObjCTy *first, __strong ObjCTy *second) EXT_RET {
  first = 0; // expected-error{{variable declared with 'objc_externally_retained' cannot be modified in ARC}}
  second = 0; // fine
}

void test6(ObjCTy *first,
           __strong ObjCTy *second) EXT_RET {
  first = 0; // expected-error{{variable declared with 'objc_externally_retained' cannot be modified in ARC}}
  second = 0;
}

__attribute__((objc_root_class)) @interface Y @end

@implementation Y
- (void)test7:(__strong ObjCTy *)first
    withThird:(ObjCTy *)second EXT_RET {
  first = 0;
  second = 0; // expected-error{{variable declared with 'objc_externally_retained' cannot be modified in ARC}}
}
@end

void (^blk)(ObjCTy *, ObjCTy *) =
    ^(__strong ObjCTy *first, ObjCTy *second) EXT_RET {
  first = 0;
  second = 0; // expected-error{{variable declared with 'objc_externally_retained' cannot be modified in ARC}}
};

void (^blk2)(ObjCTy *, ObjCTy *) =
    ^(__strong ObjCTy *first, ObjCTy *second) __attribute__((objc_externally_retained)) {
  first = 0;
  second = 0; // expected-error{{variable declared with 'objc_externally_retained' cannot be modified in ARC}}
};

void test8(EXT_RET ObjCTy *x) {} // expected-warning{{'objc_externally_retained' attribute only applies to variables}}

#pragma clang attribute ext_ret.push(__attribute__((objc_externally_retained)), apply_to=any(function, block, objc_method))
void test9(ObjCTy *first, __strong ObjCTy *second) {
  first = 0; // expected-error{{variable declared with 'objc_externally_retained' cannot be modified in ARC}}
  second = 0;
}
void (^test10)(ObjCTy *first, ObjCTy *second) = ^(ObjCTy *first, __strong ObjCTy *second) {
  first = 0; // expected-error{{variable declared with 'objc_externally_retained' cannot be modified in ARC}}
  second = 0;
};
__attribute__((objc_root_class)) @interface Test11 @end
@implementation Test11
-(void)meth: (ObjCTy *)first withSecond:(__strong ObjCTy *)second {
  first = 0; // expected-error{{variable declared with 'objc_externally_retained' cannot be modified in ARC}}
  second = 0;
}
+(void)othermeth: (ObjCTy *)first withSecond:(__strong ObjCTy *)second {
  first = 0; // expected-error{{variable declared with 'objc_externally_retained' cannot be modified in ARC}}
  second = 0;
}
@end

#if __cplusplus
class Test12 {
  void inline_member(ObjCTy *first, __strong ObjCTy *second) {
    first = 0; // expected-error{{variable declared with 'objc_externally_retained' cannot be modified in ARC}}
    second = 0;
  }
  static void static_inline_member(ObjCTy *first, __strong ObjCTy *second) {
    first = 0; // expected-error{{variable declared with 'objc_externally_retained' cannot be modified in ARC}}
    second = 0;
  }
};
#endif

void test13(ObjCTy *first, __weak ObjCTy *second, __unsafe_unretained ObjCTy *third, __strong ObjCTy *fourth) {
  first = 0; // expected-error{{variable declared with 'objc_externally_retained' cannot be modified in ARC}}
  second = 0;
  third = 0;
  fourth = 0;
}

#pragma clang attribute ext_ret.pop

__attribute__((objc_externally_retained))
void unprototyped();
