// RUN: %clang -S -emit-llvm -m64 -fobjc-abi-version=2 %s -o /dev/null

typedef unsigned int UInt_t;

@interface A
{
@protected
  UInt_t _f1;
}
@end

@interface B : A { }
@end

@interface A ()
@property (assign) UInt_t f1;
@end

@interface B ()
@property (assign) int x;
@end

@implementation B
@synthesize x;
- (id) init
{
  _f1 = 0;
  return self;
}
@end
