// RUN: %clang_cc1 %s -fblocks -fsyntax-only -verify

#define ASYNC(...) __attribute__((swift_async(__VA_ARGS__)))
#define ASYNC_ERROR(...) __attribute__((swift_async_error(__VA_ARGS__)))

ASYNC(swift_private, 1)
ASYNC_ERROR(zero_argument, 1)
void test_good(void (^handler)(int));

ASYNC(swift_private, 2)
ASYNC_ERROR(nonzero_argument, 2)
void test_good2(double, void (^handler)(double, int, double));

enum SomeEnum { SE_a, SE_b };

ASYNC(swift_private, 1)
ASYNC_ERROR(nonzero_argument, 1)
void test_good3(void (^handler)(enum SomeEnum, double));

ASYNC_ERROR(zero_argument, 1)
ASYNC(swift_private, 1)
void test_rev_order(void (^handler)(int));

@class NSError;

ASYNC(swift_private, 1)
ASYNC_ERROR(nonnull_error)
void test_nserror(void (^handler)(NSError *));

typedef struct __attribute__((objc_bridge(NSError))) __CFError * CFErrorRef;

ASYNC(swift_private, 1)
ASYNC_ERROR(nonnull_error)
void test_cferror(void (^handler)(CFErrorRef));

ASYNC(swift_private, 1)
ASYNC_ERROR(nonnull_error) // expected-error {{'swift_async_error' attribute with 'nonnull_error' convention can only be applied to a function with a completion handler with an error parameter}}
void test_interror(void (^handler)(int));

ASYNC(swift_private, 1)
ASYNC_ERROR(zero_argument, 1) // expected-error {{'swift_async_error' attribute with 'zero_argument' convention must have an integral-typed parameter in completion handler at index 1, type here is 'double'}}
void test_not_integral(void (^handler)(double));

ASYNC(swift_private, 1)
ASYNC_ERROR(none)
void test_none(void (^)(void));

ASYNC(none)
ASYNC_ERROR(none)
void test_double_none(void (^)(void));

ASYNC(none)
ASYNC_ERROR(none, 1) // expected-error {{'swift_async_error' attribute takes one argument}}
void test_double_none_args(void);

ASYNC(swift_private, 1)
ASYNC_ERROR(nonnull_error, 1) // expected-error{{'swift_async_error' attribute takes one argument}}
void test_args(void (^)(void));

ASYNC(swift_private, 1)
ASYNC_ERROR(zero_argument, 1, 1) // expected-error{{'swift_async_error' attribute takes no more than 2 arguments}}
void test_args2(void (^)(int));

ASYNC_ERROR(none) int x; // expected-warning{{'swift_async_error' attribute only applies to functions and Objective-C methods}}

@interface ObjC
-(void)m1:(void (^)(int))handler
  ASYNC(swift_private, 1)
  ASYNC_ERROR(zero_argument, 1);

-(void)m2:(int)first withSecond:(void (^)(int))handler
  ASYNC(swift_private, 2)
  ASYNC_ERROR(nonzero_argument, 1);

-(void)m3:(void (^)(void))block
  ASYNC_ERROR(zero_argument, 1) // expected-error {{'swift_async_error' attribute parameter 2 is out of bounds}}
  ASYNC(swift_private, 1);

-(void)m4:(void (^)(double, int, float))handler
  ASYNC(swift_private, 1)
  ASYNC_ERROR(nonzero_argument, 1); // expected-error{{swift_async_error' attribute with 'nonzero_argument' convention must have an integral-typed parameter in completion handler at index 1, type here is 'double'}}

-(void)m5:(void (^)(NSError *))handler
  ASYNC(swift_private, 1)
  ASYNC_ERROR(nonnull_error);

-(void)m6:(void (^)(void *))handler
  ASYNC(swift_private, 1)
  ASYNC_ERROR(nonnull_error); // expected-error{{'swift_async_error' attribute with 'nonnull_error' convention can only be applied to a method with a completion handler with an error parameter}}
@end

// 'swift_error' and 'swift_async_error' are OK on one function.
ASYNC(swift_private, 1)
ASYNC_ERROR(nonnull_error)
__attribute__((swift_error(nonnull_error)))
void swift_error_and_swift_async_error(void (^handler)(NSError *), NSError **);

@interface TestNoSwiftAsync
// swift_async_error can make sense without swift_async.
-(void)doAThingWithCompletion:(void (^)(NSError *))completion
  ASYNC_ERROR(nonnull_error);
@end
