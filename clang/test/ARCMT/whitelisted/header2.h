
#define NS_ENUM(_type, _name) enum _name : _type _name; enum _name : _type
typedef enum : NSInteger {five} ApplicableEnum;

@interface I2 : NSObject
-(int)prop;
-(void)setProp:(int)p;
@end
