// RUN: %clang_cc1 -emit-llvm -o %t %s

@interface Object
- (id) new;
@end

@interface SomeClass : Object
{
  int _myValue;
}
@property int myValue;
@end

@implementation SomeClass
@synthesize myValue=_myValue;
@end

int main(void)
{
    int val;
    SomeClass *o = [SomeClass new];
    o.myValue = -1;
    val = o.myValue++; /* val -1, o.myValue 0 */
    val += o.myValue--; /* val -1. o.myValue -1 */
    val += ++o.myValue; /* val -1, o.myValue 0 */
    val += --o.myValue; /* val -2, o.myValue -1 */
    return ++o.myValue + (val+2);
}

