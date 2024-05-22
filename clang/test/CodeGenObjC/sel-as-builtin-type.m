// RUN: %clang_cc1 -emit-llvm -o %t %s
// pr5025

typedef const struct objc_selector {
  void *sel_id;
  const char *sel_types;
} *SEL;

@interface I2
+(id) dictionary;
@end

@implementation I3; // expected-warning {{cannot find interface declaration for 'I3'}}
+(void) initialize {
  I2 *a0 = [I2 dictionary];
}
@end

int func(SEL s1, SEL s2)
{
        return s1->sel_id == s2->sel_id;
}
