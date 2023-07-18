// RUN: %clang_cc1 -emit-llvm -o %t %s
// rdar: // 8062778

@interface NSDictionary @end

@interface NSMutableDictionary : NSDictionary
@end

@interface MutableMyClass 
- (NSMutableDictionary *)myDict;
- (void)setMyDict:(NSDictionary *)myDict;

- (NSMutableDictionary *)myLang;
- (void)setMyLang:(NSDictionary *)myLang;
@end

@interface AnotherClass @end

@implementation AnotherClass
- (void)foo
{
    MutableMyClass * myObject;
    NSDictionary * newDict;
    myObject.myDict = newDict; 
    myObject.myLang = newDict;
}
@end
