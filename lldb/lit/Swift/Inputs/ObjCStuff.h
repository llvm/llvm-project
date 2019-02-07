#import <Foundation/Foundation.h>

enum CMYK { cyan, magenta, yellow, black };

typedef enum CMYK FourColors;

union Union { int i; };

@interface ObjCClass : NSNumber
- (NSString *)debugDescription;
@end

