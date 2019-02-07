@import ObjCModule;

@implementation ObjCClass {
  int private_ivar;
}

- (instancetype)init {
  self = [super init];
  if (self) {
    private_ivar = 42;
  }
  return self;
}
@end
