// RUN: %clang_cc1 -triple i386-apple-darwin9 -fobjc-runtime=macosx-fragile-10.5 -emit-llvm -o %t %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin9 -fobjc-runtime=macosx-fragile-10.5 -emit-llvm -o %t %s

@interface I0 {
@public
  _Complex float iv0;
}

@property(assign) _Complex float p0;

-(_Complex float) im0;
-(void) setIm0: (_Complex float) a0;
@end

@implementation I0 
@dynamic p0;

-(id) init {
  self->iv0 = 5.0 + 2.0i;
  return self;
}

-(_Complex float) im0 {
  printf("im0: %.2f + %.2fi\n", __real iv0, __imag iv0);
  return iv0 + (.1 + .2i);
}
-(void) setIm0: (_Complex float) a0 {
  printf("setIm0: %.2f + %.2fi\n", __real a0, __imag a0);
  iv0 = a0 + (.3 + .4i);
}

-(_Complex float) p0 {
  printf("p0: %.2f + %.2fi\n", __real iv0, __imag iv0);
  return iv0 + (.5 + .6i);
}
-(void) setP0: (_Complex float) a0 {
  printf("setP0: %.2f + %.2fi\n", __real a0, __imag a0);
  iv0 = a0 + (.7 + .8i);
}
@end

void f0(I0 *a0) {
    float l0 = __real a0.im0;
    float l1 = __imag a0->iv0;
    _Complex float l2 = (a0.im0 = a0.im0);
    _Complex float l3 = a0->iv0;
    _Complex float l4 = (a0->iv0 = a0->iv0);
    _Complex float l5 = a0->iv0;
    _Complex float l6 = (a0.p0 = a0.p0);
    _Complex float l7 = a0->iv0;
    _Complex float l8 = [a0 im0];
    printf("l0: %.2f + %.2fi\n", __real l0, __imag l0);
    printf("l1: %.2f + %.2fi\n", __real l1, __imag l1);
    printf("l2: %.2f + %.2fi\n", __real l2, __imag l2);
    printf("l3: %.2f + %.2fi\n", __real l3, __imag l3);
    printf("l4: %.2f + %.2fi\n", __real l4, __imag l4);
    printf("l5: %.2f + %.2fi\n", __real l5, __imag l5);
    printf("l6: %.2f + %.2fi\n", __real l6, __imag l6);
    printf("l7: %.2f + %.2fi\n", __real l7, __imag l7);
    printf("l8: %.2f + %.2fi\n", __real l8, __imag l8);
}
