#pragma clang system_header

#define nil 0
#define BOOL int

#define NS_ASSUME_NONNULL_BEGIN _Pragma("clang assume_nonnull begin")
#define NS_ASSUME_NONNULL_END   _Pragma("clang assume_nonnull end")

NS_ASSUME_NONNULL_BEGIN

typedef struct _NSZone NSZone;
typedef unsigned long NSUInteger;
@class NSCoder, NSEnumerator;

@protocol NSObject
+ (instancetype)alloc;
- (instancetype)init;
- (instancetype)autorelease;
@end

@protocol NSCopying
- (id)copyWithZone:(nullable NSZone *)zone;
@end

@protocol NSMutableCopying
- (id)mutableCopyWithZone:(nullable NSZone *)zone;
@end

@protocol NSCoding
- (void)encodeWithCoder:(NSCoder *)aCoder;
@end

@protocol NSSecureCoding <NSCoding>
@required
+ (BOOL)supportsSecureCoding;
@end

typedef struct {
  unsigned long state;
  id *itemsPtr;
  unsigned long *mutationsPtr;
  unsigned long extra[5];
} NSFastEnumerationState;

__attribute__((objc_root_class))
@interface
NSObject<NSObject>
@end

@interface NSString : NSObject<NSCopying>
- (BOOL)isEqualToString : (NSString *)aString;
- (NSString *)stringByAppendingString:(NSString *)aString;
- (nullable NSString *)nullableStringByAppendingString:(NSString *)aString;
+ (NSString * _Nonnull) generateString;
+ (NSString *) generateImplicitlyNonnullString;
+ (NSString * _Nullable) generatePossiblyNullString;
@end

void NSSystemFunctionTakingNonnull(NSString *s);

@interface NSSystemClass : NSObject
- (void) takesNonnull:(NSString *)s;
@end

NSString* _Nullable getPossiblyNullString();
NSString* _Nonnull  getString();

@protocol MyProtocol
- (NSString * _Nonnull) getString;
@end

NS_ASSUME_NONNULL_END

@interface NSDictionary : NSObject <NSCopying, NSMutableCopying, NSSecureCoding>

- (NSUInteger)count;
- (id)objectForKey:(id)aKey;
- (NSEnumerator *)keyEnumerator;
- (id)objectForKeyedSubscript:(id)aKey;

@end

@interface NSDictionary (NSDictionaryCreation)

+ (id)dictionary;
+ (id)dictionaryWithObject:(id)object forKey:(id <NSCopying>)key;
+ (instancetype)dictionaryWithObjects:(const id [])objects forKeys:(const id <NSCopying> [])keys count:(NSUInteger)cnt;

@end

@interface NSMutableDictionary : NSDictionary

- (void)removeObjectForKey:(id)aKey;
- (void)setObject:(id)anObject forKey:(id <NSCopying>)aKey;

@end

@interface NSMutableDictionary (NSExtendedMutableDictionary)

- (void)addEntriesFromDictionary:(NSDictionary *)otherDictionary;
- (void)removeAllObjects;
- (void)setDictionary:(NSDictionary *)otherDictionary;
- (void)setObject:(id)obj forKeyedSubscript:(id <NSCopying>)key __attribute__((availability(macosx,introduced=10.8)));

@end
