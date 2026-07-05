#ifndef SOMEKIT_H
#define SOMEKIT_H

__attribute__((objc_root_class))
#ifndef NS_ASSUME_NONNULL_BEGIN
#if __has_feature(assume_nonnull)
#define NS_ASSUME_NONNULL_BEGIN _Pragma("clang assume_nonnull begin")
#define NS_ASSUME_NONNULL_END _Pragma("clang assume_nonnull end")
#else
#define NS_ASSUME_NONNULL_BEGIN
#define NS_ASSUME_NONNULL_END
#endif
#endif

NS_ASSUME_NONNULL_BEGIN

@interface A
-(null_unspecified A*)transform:(null_unspecified A*)input __attribute__((unavailable("anything but this")));
-(A*)transform:(A*)input integer:(int)integer;

@property (null_unspecified, nonatomic, readonly, retain) A* someA;
@property (null_unspecified, nonatomic, retain) A* someOtherA;

@property (nonatomic) int intValue __attribute__((unavailable("wouldn't work anyway")));
@end

NS_ASSUME_NONNULL_END


__attribute__((unavailable("just don't")))
@interface B : A
@end

@interface C : A
- (instancetype)init; // expected-warning{{pointer is missing a nullability type specifier}}
// expected-note@-1{{insert '_Nullable' if the pointer may be null}}
// expected-note@-2{{insert '_Nonnull' if the pointer should never be null}}
- (instancetype)initWithA:( A*)a __attribute__((objc_designated_initializer)); // expected-warning 2{{pointer is missing a nullability type specifier}}
// expected-note@-1 2{{insert '_Nullable' if the pointer may be null}}
// expected-note@-2 2{{insert '_Nonnull' if the pointer should never be null}}
@end

#endif

