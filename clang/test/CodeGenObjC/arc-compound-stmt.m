// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -fobjc-arc -o - %s

typedef unsigned long NSUInteger;

@interface NSString
- (NSString *)stringByAppendingString:(NSString *)aString;
- (NSString *)substringFromIndex:(NSUInteger)from;
@end

@interface MyClass
- (void)inst;
@end

@implementation MyClass

- (void)inst;
{
    NSString *propName;

    NSString *capitalPropName = ({
        NSString *cap;
        if (propName)
            cap = [cap stringByAppendingString:[propName substringFromIndex:1]];
        cap;
    });
}

@end
