@import ObjCModule;

@interface ProtoImpl : NSObject <ObjCProtocol>
- (NSString *)name;
@end

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

id<ObjCProtocol> getProto() {
  return [ProtoImpl new];
}

@implementation ProtoImpl
- (NSString *)name {
  return @"I am implementing an Objective-C protocol.";
}
@end

