// RUN: %clang_cc1  -fsyntax-only -verify %s
// RUN: %clang_cc1 -x objective-c++  -fsyntax-only -verify %s

#if __LP64__ || (TARGET_OS_EMBEDDED && !TARGET_OS_IPHONE) || TARGET_OS_WIN32 || NS_BUILD_32_LIKE_64
typedef unsigned long NSUInteger;
#else
typedef unsigned int NSUInteger;
#endif

@protocol NSObject
@end

@protocol NSCopying
@end

__attribute__((objc_root_class))
@interface NSObject <NSObject>
@end

@interface NSString : NSObject <NSCopying>
@end

@interface NSNumber : NSObject <NSCopying>
+ (NSNumber *)numberWithInt:(int)value;
@end

@interface NSArray<T> : NSObject <NSCopying>
+ (instancetype)arrayWithObjects:(const T [])objects count:(NSUInteger)cnt;
@end

@interface NSDictionary<K, V> : NSObject <NSCopying>
+ (instancetype)dictionaryWithObjects:(const V [])objects forKeys:(const K [])keys count:(NSUInteger)cnt;
@end

void testArrayLiteral(void) {
  NSArray<NSString *> *array1 = @[@"hello",
                                   @1, // expected-warning{{of type 'NSNumber *' is not compatible with array element type 'NSString *'}}
                                  @"world",
                                  @[@1, @2]]; // expected-warning{{of type 'NSArray *' is not compatible with array element type 'NSString *'}}

  NSArray<NSArray<NSString *> *> *array2 = @[@[@"hello", @"world"],
                                              @"blah", // expected-warning{{object of type 'NSString *' is not compatible with array element type 'NSArray<NSString *> *'}}
                                             @[@1]]; // expected-warning{{object of type 'NSNumber *' is not compatible with array element type 'NSString *'}}
}

void testDictionaryLiteral(void) {
  NSDictionary<NSString *, NSNumber *> *dict1 = @{
    @"hello" : @17,
    @18 : @18, // expected-warning{{object of type 'NSNumber *' is not compatible with dictionary key type 'NSString *'}}
    @"world" : @"blah" // expected-warning{{object of type 'NSString *' is not compatible with dictionary value type 'NSNumber *'}}
  };
}
