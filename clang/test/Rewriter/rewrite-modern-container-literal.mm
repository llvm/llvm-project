// RUN: %clang_cc1 -x objective-c++ -fms-extensions -rewrite-objc %s -o %t-rw.cpp
// RUN: %clang_cc1 -fsyntax-only -Wno-address-of-temporary -D"id=void*" -D"SEL=void*" -D"__declspec(X)=" %t-rw.cpp
// rdar://10803676

void *sel_registerName(const char *);
typedef unsigned long NSUInteger;
typedef long NSInteger;
typedef bool BOOL;

@interface NSNumber
+ (NSNumber *)numberWithChar:(char)value;
+ (NSNumber *)numberWithUnsignedChar:(unsigned char)value;
+ (NSNumber *)numberWithShort:(short)value;
+ (NSNumber *)numberWithUnsignedShort:(unsigned short)value;
+ (NSNumber *)numberWithInt:(int)value;
+ (NSNumber *)numberWithUnsignedInt:(unsigned int)value;
+ (NSNumber *)numberWithLong:(long)value;
+ (NSNumber *)numberWithUnsignedLong:(unsigned long)value;
+ (NSNumber *)numberWithLongLong:(long long)value;
+ (NSNumber *)numberWithUnsignedLongLong:(unsigned long long)value;
+ (NSNumber *)numberWithFloat:(float)value;
+ (NSNumber *)numberWithDouble:(double)value;
+ (NSNumber *)numberWithBool:(BOOL)value;
+ (NSNumber *)numberWithInteger:(NSInteger)value ;
+ (NSNumber *)numberWithUnsignedInteger:(NSUInteger)value ;
@end

@protocol NSCopying @end

@interface NSDictionary
+ (id)dictionaryWithObjects:(const id [])objects forKeys:(const id <NSCopying> [])keys count:(NSUInteger)cnt;
@end

@interface NSArray 
+ (id)arrayWithObjects:(const id [])objects count:(NSUInteger)cnt;
@end

@interface NSString<NSCopying>
@end

id NSUserName();

@interface NSDate
+ (id)date;
@end

int main() {
NSArray *array = @[ @"Hello", NSUserName(), [NSDate date], [NSNumber numberWithInt:42]];

NSDictionary *dictionary = @{ @"name" : NSUserName(), @"date" : [NSDate date], @"process" : @"processInfo"};

NSDictionary *dict = @{ @"name":@666, @"man":@__objc_yes, @"date":@1.3 };

}

