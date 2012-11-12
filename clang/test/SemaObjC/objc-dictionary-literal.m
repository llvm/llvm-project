// RUN: %clang_cc1  -fsyntax-only -verify %s
// rdar://11062080

@interface NSNumber
+ (NSNumber *)numberWithChar:(char)value;
+ (NSNumber *)numberWithInt:(int)value;
@end

@protocol NSCopying @end
typedef unsigned long NSUInteger;
typedef long NSInteger;

@interface NSDictionary
+ (id)dictionaryWithObjects:(const id [])objects forKeys:(const id <NSCopying> [])keys count:(NSUInteger)cnt;
- (void)setObject:(id)object forKeyedSubscript:(id)key;
@end

@interface NSString<NSCopying>
@end

@interface NSArray
- (id)objectAtIndexedSubscript:(NSInteger)index;
- (void)setObject:(id)object atIndexedSubscript:(NSInteger)index;
@end

int main() {
	NSDictionary *dict = @{ @"name":@666 };
        dict[@"name"] = @666;

        dict["name"] = @666; // expected-error {{indexing expression is invalid because subscript type 'char *' is not an Objective-C pointer}}

	return 0;
}

