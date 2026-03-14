// RUN: %clang_cc1 -triple x86_64-apple-darwin10  -emit-llvm -o - %s | FileCheck -check-prefix CHECK-LP64 %s

@interface A
@property __complex int COMPLEX_PROP;
- (__complex int)y;
- (void) setY : (__complex int)rhs;
@end

void f0(A *a) {  
  _Complex int a1 = 25 + 10i;
  a.COMPLEX_PROP += a1;
  a.y += a1;
}

// CHECK-LP64: private unnamed_addr constant [13 x i8] c"COMPLEX_PROP
// CHECK-LP64: private unnamed_addr constant [17 x i8] c"setCOMPLEX_PROP

@interface B
@property (assign) _Complex float f_complex_ivar;
@end

@implementation B

@synthesize f_complex_ivar = _f_complex_ivar;
-(void) unary_f_complex: (_Complex float) a0 {
  self.f_complex_ivar = a0;
}

@end

