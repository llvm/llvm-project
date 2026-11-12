// RUN: %check_clang_tidy %s bugprone-assert-side-effect %t

int abort(void);

@interface NSObject
@end

@interface NSString
@end

@interface NSAssertionHandler
+ (NSAssertionHandler *)currentHandler;
- handleFailureInMethod:(SEL)cmd object:(NSObject *)obj desc:(NSString *)desc;
- handleFailureInFunction:(NSString *)desc;
@end

#ifndef NDEBUG
#define NSAssert(condition, description, ...)                                    \
  do {                                                                           \
    if (__builtin_expect(!(condition), 0)) {                                     \
      [[NSAssertionHandler currentHandler] handleFailureInMethod:_cmd            \
                                                          object:self            \
                                                            desc:(description)]; \
    }                                                                            \
  } while (0);
#define NSCAssert(condition, description, ...)                                     \
  do {                                                                             \
    if (__builtin_expect(!(condition), 0)) {                                       \
      [[NSAssertionHandler currentHandler] handleFailureInFunction:(description)]; \
    }                                                                              \
  } while (0);
#else
#define NSAssert(condition, description, ...) do {} while (0)
#define NSCAssert(condition, description, ...) do {} while (0)
#endif

@interface I : NSObject
- (void)foo;
@end

@implementation I
- (void)foo {
  int x = 0;
  NSAssert((++x) == 1, @"Ugh.");
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: side effect in NSAssert() condition discarded in release builds [bugprone-assert-side-effect]
}
@end

void foo(void) {
  int x = 0;
  NSCAssert((++x) == 1, @"Ugh.");
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: side effect in NSCAssert() condition discarded in release builds [bugprone-assert-side-effect]
}
