// RUN: %clang_cc1 -x objective-c -triple x86_64-apple-darwin10 -fblocks -emit-llvm %s -o /dev/null
// RUN: %clang_cc1 -x objective-c++ -triple x86_64-apple-darwin10 -fblocks -emit-llvm %s -o /dev/null

@interface NSNumber
+ (NSNumber *)numberWithChar:(char)value;
+ (NSNumber *)numberWithInt:(int)value;
@end

@protocol NSCopying @end
typedef unsigned long NSUInteger;

@interface NSDictionary
+ (id)dictionaryWithObjects:(const id [])objects forKeys:(const id <NSCopying> [])keys count:(NSUInteger)cnt;
@end

@interface NSString<NSCopying>
@end

int main(void) {
	NSDictionary *dict = @{ @"name":@666 };
	NSDictionary *dict1 = @{ @"name":@666 };
	NSDictionary *dict2 = @{ @"name":@666 };
	return 0;
}
