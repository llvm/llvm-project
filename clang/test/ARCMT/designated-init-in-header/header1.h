#define NS_DESIGNATED_INITIALIZER __attribute__((objc_designated_initializer))

@class NSString;

@interface B1
-(id)init;
@end

@interface S1 : B1
-(int)prop;
-(void)setProp:(int)p;
+(id)s1;
-(id)initWithFoo:(NSString*)foo;
@end
