@interface NSSomeClass
-(instancetype)init;
@end

@interface NSSomeClass (SomeCategory)
- (void)methodWithRawName:(int)x;
@end

enum NSSomeEnum {
  NSSomeEnumWithRed,
  NSSomeEnumWithGreen,
  NSSomeEnumWithBlue,
  NSSomeEnumWithRawName,
};
