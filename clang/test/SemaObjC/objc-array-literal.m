// RUN: %clang_cc1  -fsyntax-only -verify %s
// rdar://10111397

#if __LP64__ || (TARGET_OS_EMBEDDED && !TARGET_OS_IPHONE) || TARGET_OS_WIN32 || NS_BUILD_32_LIKE_64
typedef unsigned long NSUInteger;
#else
typedef unsigned int NSUInteger;
#endif

@class NSString;

extern void NSLog(NSString *format, ...) __attribute__((format(__NSString__, 1, 2)));

@class NSFastEnumerationState;

@protocol NSFastEnumeration

- (NSUInteger)countByEnumeratingWithState:(NSFastEnumerationState *)state objects:(id [])buffer count:(NSUInteger)len;

@end

@interface NSNumber 
+ (NSNumber *)numberWithInt:(int)value;
@end

@interface NSArray <NSFastEnumeration>
+ (id)arrayWithObjects:(const id [])objects count:(NSUInteger)cnt;
@end


int main() {
 NSArray *array = @[@"Hello", @"There", @"How Are You", [NSNumber numberWithInt:42]];

  for (id string in array)
    NSLog(@"%@\n", string);

  NSArray *array1 = @["Forgot"]; // expected-error {{string literal must be prefixed by '@' in a collection}}

  const char *blah;
  NSArray *array2 = @[blah]; // expected-error{{collection element of type 'const char *' is not an Objective-C object}}
}
