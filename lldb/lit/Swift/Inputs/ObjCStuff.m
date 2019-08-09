@import ObjCStuff;

@implementation ObjCClass
- (id)init {
  self = [NSNumber numberWithInt: 1234];
  return self;
}
- (NSString * _Nonnull)debugDescription {
  return @"Hello from Objective-C!";
}
@end;

const MyFloat globalFloat = 3.14f;
