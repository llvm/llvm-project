struct RefCountedType { int value; };

@interface MethodTest
- (struct RefCountedType *)getUnowned;
- (struct RefCountedType *)getOwned;
@end

struct RefCountedType * getObjCUnowned(void);
struct RefCountedType * getObjCOwned(void);
