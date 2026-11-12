@interface Object
@end

@interface SomeBaseClass : Object
+ (nullable instancetype)instancetypeFactoryMethod;
+ (nullable SomeBaseClass *)staticFactoryMethod;
@end

@interface SomeSubclass : SomeBaseClass
@end
