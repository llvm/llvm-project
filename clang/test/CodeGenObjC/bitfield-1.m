// RUN: %clang_cc1 -triple x86_64-apple-darwin9 -fobjc-runtime=macosx-fragile-10.5 -emit-llvm -o %t %s
// RUN: %clang_cc1 -triple i386-apple-darwin9 -fobjc-runtime=macosx-fragile-10.5 -emit-llvm -o %t %s
// RUN: %clang_cc1 -triple i386-pc-linux-gnu -fobjc-runtime=macosx-fragile-10.5 -emit-llvm -o %t %s

@interface Object
- (id) alloc;
- (id) init;
@end

extern void abort(void);

#define CHECK_IF(expr) if(!(expr)) abort();

@interface Base: Object 
{
    int full;
    int full2: 32;
    int _refs: 8;
    int field2: 3;
    unsigned f3: 8;
    short cc;
    unsigned g: 16;
    int r2: 8;
    int r3: 8;
    int r4: 2;
    int r5: 8;
    char c;
}
- (void)setValues;
@end

@interface Derived: Base
{
    char d;
    int _field3: 6;
}
- (void)checkValues;
@end

@implementation Base
-(void)setValues {
  full = 1;
  full2 = 2;
  _refs = 3;
  field2 = 1;
  f3 = 6;
  cc = 7;
  g = 8;
  r2 = 9;
  r3 = 10;
  r4 = 1;
  r5 = 12;
  c = 13;
}
@end

@implementation Derived
-(void)checkValues {
  CHECK_IF(full == 1);
  CHECK_IF(full2 == 2);
  CHECK_IF(_refs == 3);
  CHECK_IF(field2 == 1);
  CHECK_IF(f3 == 6);
  CHECK_IF(cc == 7);
  CHECK_IF(g == 8);
  CHECK_IF(r2 == 9);
  CHECK_IF(r3 == 10);
  CHECK_IF(r4 == 1);
  CHECK_IF(r5 == 12);
  CHECK_IF(c == 13);
}
@end

int main(void) {
  Derived *obj = [[Derived alloc] init];

  [obj setValues];
  [obj checkValues];

  return 0;
}
