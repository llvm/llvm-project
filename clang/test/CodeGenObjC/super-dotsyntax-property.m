// RUN: %clang_cc1 -emit-llvm -o %t %s

@interface B
{
  int _parent;
}
@property int parent;
  +(int) classGetter;
  +(void) setClassGetter:(int) arg;

  -(int) getter;
  -(void) setGetter:(int)arg;
@end

@interface A : B
@end

@implementation A
+(int) classGetter {
  return 0;
}

+(int) classGetter2 {
  super.classGetter = 100;
  return super.classGetter;
}

-(void) method {
  super.getter = 200;
  int x = super.getter;
}
-(void) setParent : (int) arg {
  super.parent = arg + super.parent;
  
}
@end

void f0() {
  int l1 = A.classGetter;
  int l2 = [A classGetter2];
}
