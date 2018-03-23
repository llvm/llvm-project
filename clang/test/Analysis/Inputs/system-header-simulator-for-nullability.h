#pragma clang system_header

#define nil 0
#define BOOL int

#define NS_ASSUME_NONNULL_BEGIN _Pragma("clang assume_nonnull begin")
#define NS_ASSUME_NONNULL_END   _Pragma("clang assume_nonnull end")

NS_ASSUME_NONNULL_BEGIN

typedef struct _NSZone NSZone;

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

__attribute__((objc_root_class))
@interface
NSObject<NSObject>
@end

@interface NSString : NSObject<NSCopying>
- (BOOL)isEqualToString : (NSString *)aString;
- (NSString *)stringByAppendingString:(NSString *)aString;
+ (_Nonnull NSString *) generateString;
+ (_Nullable NSString *) generatePossiblyNullString;
@end

void NSSystemFunctionTakingNonnull(NSString *s);

@interface NSSystemClass : NSObject
- (void) takesNonnull:(NSString *)s;
@end

NSString* _Nullable getPossiblyNullString();
NSString* _Nonnull  getString();

@protocol MyProtocol
- (_Nonnull NSString *) getString;
@end

NS_ASSUME_NONNULL_END
